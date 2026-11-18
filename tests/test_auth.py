import json
import threading
import urllib.error
import urllib.request

import pytest

from agent_project import auth as auth_module
from agent_project.auth import Account, Auth, LockedOut
from agent_project.llm import MockClient
from agent_project.web import make_server

PASSWORD = "dobre-haslo-123"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def account() -> Account:
    return Account.create("wiktor", PASSWORD)


def test_password_is_stored_only_as_a_hash(account, tmp_path) -> None:
    path = tmp_path / "account.json"
    account.save(path)

    text = path.read_text("utf-8")
    assert PASSWORD not in text
    loaded = Account.load(path)
    assert loaded.check("wiktor", PASSWORD)
    assert not loaded.check("wiktor", PASSWORD + "x")
    assert not loaded.check("ktos", PASSWORD)


def test_short_password_is_rejected() -> None:
    with pytest.raises(ValueError, match="Hasło"):
        Account.create("wiktor", "krotkie")


def test_session_expires_and_logout_ends_it(account) -> None:
    clock = Clock()
    auth = Auth(account, session_seconds=60, clock=clock)
    token = auth.login("wiktor", PASSWORD)

    assert auth.valid(token)
    assert not auth.valid("zmyslony-token")
    clock.now += 61
    assert not auth.valid(token)

    token = auth.login("wiktor", PASSWORD)
    auth.logout(token)
    assert not auth.valid(token)


def test_too_many_wrong_passwords_lock_the_login(account) -> None:
    clock = Clock()
    auth = Auth(account, clock=clock)
    for _ in range(auth_module.MAX_FAILURES):
        assert auth.login("wiktor", "zle-haslo-000") is None

    # Even the right password waits until the lockout ends.
    with pytest.raises(LockedOut):
        auth.login("wiktor", PASSWORD)
    clock.now += auth_module.LOCKOUT_SECONDS + 1
    assert auth.login("wiktor", PASSWORD) is not None


# --- the web app behind the login -------------------------------------------


@pytest.fixture
def app(account, tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "profiles").mkdir(parents=True)
    (data_dir / "job_offers.json").write_text("[]", encoding="utf-8")
    httpd = make_server(
        port=0,
        data_dir=data_dir,
        client_factory=lambda: MockClient([]),
        auth=Auth(account),
        secure_cookie=False,
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def call(url, method="GET", body=None, cookie=None, headers=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if cookie:
        req.add_header("Cookie", cookie)
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(req, timeout=10) as res:
            return res.status, res.headers, res.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def login(base) -> str:
    status, headers, _ = call(
        base + "/api/login", "POST", {"username": "wiktor", "password": PASSWORD}
    )
    assert status == 200
    set_cookie = headers["Set-Cookie"]
    assert "HttpOnly" in set_cookie and "SameSite=Strict" in set_cookie
    return set_cookie.split(";")[0]


def test_pages_and_api_need_a_login(app) -> None:
    status, headers, _ = call(app + "/")
    assert status == 303 and headers["Location"] == "/login"

    status, _, body = call(app + "/api/profiles")
    assert status == 401

    status, _, body = call(app + "/api/chat/reset", "POST", {})
    assert status == 401

    status, _, body = call(app + "/login")
    assert status == 200 and b'id="password"' in body


def test_wrong_password_is_refused(app) -> None:
    status, headers, body = call(
        app + "/api/login", "POST", {"username": "wiktor", "password": "nie-to-haslo"}
    )
    assert status == 401
    assert "Set-Cookie" not in headers


def test_login_opens_the_app_and_logout_closes_it(app) -> None:
    cookie = login(app)

    status, headers, body = call(app + "/", cookie=cookie)
    assert status == 200 and b"chat-input" in body
    assert headers["X-Frame-Options"] == "DENY"
    status, _, _ = call(app + "/api/profiles", cookie=cookie)
    assert status == 200

    call(app + "/api/logout", "POST", {}, cookie=cookie)
    status, _, _ = call(app + "/api/profiles", cookie=cookie)
    assert status == 401


def test_posts_from_another_site_are_refused(app) -> None:
    cookie = login(app)

    status, _, _ = call(
        app + "/api/chat/reset",
        "POST",
        {},
        cookie=cookie,
        headers={"Origin": "https://evil.example"},
    )
    assert status == 403

    # A plain form post (not JSON) is refused too.
    req = urllib.request.Request(app + "/api/chat/reset", data=b"a=1", method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Cookie", cookie)
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req, timeout=10)
    assert error.value.code == 403
