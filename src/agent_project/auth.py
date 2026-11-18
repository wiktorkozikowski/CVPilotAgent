"""The app's only account: password check, sessions and the login lockout.

There is no registration. The account is created (or its password changed) on
the server with

    uv run python -m agent_project.auth

which asks for a login and a password and writes var/account.json. The file
holds only a scrypt hash of the password, never the password itself.

Sessions live in memory: a restart of the app logs the user out. The browser
gets a random token in an HttpOnly, SameSite=Strict cookie; the app keeps only
the token's SHA-256.
"""

import argparse
import getpass
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from offer_server.paths import VAR_DIR

ACCOUNT_FILE = VAR_DIR / "account.json"
COOKIE_NAME = "cvpilot_session"
SESSION_SECONDS = 12 * 3600
MIN_PASSWORD_CHARS = 10
MAX_PASSWORD_CHARS = 200
# After this many wrong passwords in a row, logging in is blocked for a while.
MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def account_path() -> Path:
    value = os.getenv("CVPILOT_ACCOUNT_FILE")
    return Path(value) if value else ACCOUNT_FILE


def _hash(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=64 * 1024 * 1024
    )


@dataclass(frozen=True)
class Account:
    username: str
    salt: bytes
    password_hash: bytes
    n: int = SCRYPT["n"]
    r: int = SCRYPT["r"]
    p: int = SCRYPT["p"]

    @classmethod
    def create(cls, username: str, password: str) -> "Account":
        username = username.strip()
        if not username:
            raise ValueError("Login nie może być pusty.")
        if not MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS:
            raise ValueError(
                f"Hasło musi mieć od {MIN_PASSWORD_CHARS} do "
                f"{MAX_PASSWORD_CHARS} znaków."
            )
        salt = secrets.token_bytes(16)
        return cls(username, salt, _hash(password, salt, **SCRYPT))

    def check(self, username: str, password: str) -> bool:
        if len(password) > MAX_PASSWORD_CHARS:
            return False
        # The hash is computed even for a wrong login, so timing tells nothing.
        given = _hash(password, self.salt, self.n, self.r, self.p)
        same_user = hmac.compare_digest(
            username.strip().encode("utf-8"), self.username.encode("utf-8")
        )
        return hmac.compare_digest(given, self.password_hash) and same_user

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "username": self.username,
            "salt": self.salt.hex(),
            "password_hash": self.password_hash.hex(),
            "scrypt": {"n": self.n, "r": self.r, "p": self.p},
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> "Account":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            username=data["username"],
            salt=bytes.fromhex(data["salt"]),
            password_hash=bytes.fromhex(data["password_hash"]),
            **data.get("scrypt", SCRYPT),
        )


class LockedOut(Exception):
    def __init__(self, seconds: int) -> None:
        minutes = seconds // 60 + 1
        super().__init__(f"Za dużo nieudanych prób. Spróbuj za {minutes} min.")
        self.seconds = seconds


class Auth:
    """Checks logins for the one account and keeps the open sessions."""

    def __init__(
        self,
        account: Account,
        session_seconds: int = SESSION_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.account = account
        self.session_seconds = session_seconds
        self._clock = clock
        self._sessions: dict[str, float] = {}  # sha256(token) -> expiry
        self._failures = 0
        self._locked_until = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def login(self, username: str, password: str) -> str | None:
        """A new session token, or None for a wrong login or password.

        Raises LockedOut after MAX_FAILURES wrong attempts in a row.
        """
        with self._lock:
            now = self._clock()
            if now < self._locked_until:
                raise LockedOut(int(self._locked_until - now))
        ok = self.account.check(username, password)
        with self._lock:
            now = self._clock()
            if not ok:
                self._failures += 1
                if self._failures >= MAX_FAILURES:
                    self._failures = 0
                    self._locked_until = now + LOCKOUT_SECONDS
                return None
            self._failures = 0
            self._sessions = {k: v for k, v in self._sessions.items() if v > now}
            token = secrets.token_urlsafe(32)
            self._sessions[self._key(token)] = now + self.session_seconds
            return token

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            expiry = self._sessions.get(self._key(token))
            if expiry is None:
                return False
            if expiry <= self._clock():
                del self._sessions[self._key(token)]
                return False
            return True

    def logout(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._sessions.pop(self._key(token), None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zakłada jedyne konto aplikacji albo zmienia jego hasło."
    )
    parser.add_argument("--file", type=Path, default=account_path())
    parser.add_argument("--user", help="login (bez tej opcji program zapyta)")
    args = parser.parse_args()

    if args.file.exists():
        print(f"Konto już istnieje ({args.file}); nowe dane je zastąpią.")
    username = args.user or input("Login: ")
    password = getpass.getpass("Hasło: ")
    if getpass.getpass("Powtórz hasło: ") != password:
        sys.exit("Hasła się różnią, nic nie zapisano.")
    try:
        account = Account.create(username, password)
    except ValueError as error:
        sys.exit(str(error))
    account.save(args.file)
    print(f"Zapisano konto {account.username} w {args.file}.")
    print("Działająca aplikacja wczyta je po restarcie.")


if __name__ == "__main__":
    main()
