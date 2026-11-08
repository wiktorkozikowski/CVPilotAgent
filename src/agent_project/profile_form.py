"""Local web form for creating a user profile JSON file in data/."""

import html
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from pydantic import ValidationError

from agent_project.cv_tools import Profile, load_offers

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HOST, PORT = "127.0.0.1", 8765

LANGUAGES = [
    "polski",
    "angielski",
    "niemiecki",
    "francuski",
    "hiszpański",
    "włoski",
    "ukraiński",
    "rosyjski",
]
ROLES = [
    "Junior Data Analyst",
    "Data Engineer",
    "Machine Learning Engineer",
    "Python Developer",
    "Backend Developer",
    "Frontend Developer",
    "Mobile Developer",
    "DevOps Engineer",
    "Cybersecurity Analyst",
    "QA Engineer",
    "UX/UI Designer",
    "Project Manager",
    "Analityk biznesowy",
    "Marketing Specialist",
]

# Offer skills that are abilities or methods; everything else counts as a technology.
ABILITIES = {
    "communication",
    "copywriting",
    "english",
    "networking",
    "scripting",
    "scrum",
    "seo",
    "statistics",
    "testing",
    "ux research",
}

PAGE = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>Nowy profil</title>
<style>
body{{font-family:sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}}
h2{{margin-top:1.8rem;font-size:1.05rem}}
label.f{{display:block;margin-top:1rem;font-weight:bold}}
input[type=text],input[type=email],input[type=tel]{{width:100%;padding:.5rem;
box-sizing:border-box;font-weight:normal}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.checks{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
gap:.35rem .8rem}}
.checks label{{font-weight:normal}}
button{{margin-top:1.8rem;padding:.6rem 1.2rem}}
.msg{{margin-top:1rem;padding:.7rem;border-radius:4px}}
.ok{{background:#e6f4ea}} .err{{background:#fce8e6}}
</style></head><body>
<h1>Nowy profil użytkownika</h1>
{message}
<form method="post">
<div class="row">
<label class="f">Imię <input type="text" name="first_name" required
value="{first_name}"></label>
<label class="f">Nazwisko <input type="text" name="last_name" required
value="{last_name}"></label>
</div>
<div class="row">
<label class="f">E-mail <input type="email" name="email" value="{email}"></label>
<label class="f">Numer telefonu <input type="tel" name="phone"
value="{phone}"></label>
</div>
<label class="f">Lokalizacja <input type="text" name="location" required
value="{location}"></label>
<h2>Języki</h2><div class="checks">{languages}</div>
<h2>Docelowe role</h2><div class="checks">{target_roles}</div>
<label class="f">Inne role <small>(po przecinku)</small>
<input type="text" name="other_roles" value="{other_roles}"></label>
<h2>Umiejętności</h2><div class="checks">{skills}</div>
<label class="f">Inne umiejętności <small>(po przecinku)</small>
<input type="text" name="other_skills" value="{other_skills}"></label>
<h2>Technologie</h2><div class="checks">{technologies}</div>
<label class="f">Inne technologie <small>(po przecinku)</small>
<input type="text" name="other_technologies"
value="{other_technologies}"></label>
<button type="submit">Zapisz profil</button>
</form></body></html>"""

TEXT_FIELDS = (
    "first_name",
    "last_name",
    "email",
    "phone",
    "location",
    "other_skills",
    "other_technologies",
    "other_roles",
)
LIST_FIELDS = ("languages", "target_roles", "skills", "technologies")


def skill_choices() -> tuple[list[str], list[str]]:
    """Skills and technologies required by saved offers, so each can match an offer."""
    offers = load_offers(DATA_DIR / "job_offers.json")
    found: dict[str, str] = {}
    for offer in offers:
        for skill in offer.required_skills:
            found.setdefault(skill.casefold(), skill)
    names = sorted(found.values(), key=str.casefold)
    abilities = [n for n in names if n.casefold() in ABILITIES]
    technologies = [n for n in names if n.casefold() not in ABILITIES]
    return abilities, technologies


def split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def save_profile(form: dict[str, str | list[str]]) -> Path:
    """Validate the form values as a Profile and write them to data/."""
    skills = [
        *form["skills"],
        *split_list(str(form["other_skills"])),
        *form["technologies"],
        *split_list(str(form["other_technologies"])),
    ]
    profile = Profile(
        name=f"{str(form['first_name']).strip()} {str(form['last_name']).strip()}",
        email=str(form["email"]).strip(),
        phone=str(form["phone"]).strip(),
        location=str(form["location"]).strip(),
        target_roles=[*form["target_roles"], *split_list(str(form["other_roles"]))],
        skills=skills,
        languages=list(form["languages"]),
    )
    slug = re.sub(r"\W+", "_", profile.name.casefold()).strip("_")
    path = DATA_DIR / f"profile_{slug or 'user'}.json"
    path.write_text(
        json.dumps(profile.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def checkboxes(name: str, options: list[str], selected: list[str]) -> str:
    return "".join(
        f'<label><input type="checkbox" name="{name}" '
        f'value="{html.escape(option, quote=True)}"'
        f"{' checked' if option in selected else ''}> {html.escape(option)}</label>"
        for option in options
    )


def render(
    values: dict[str, str | list[str]] | None = None, message: str = ""
) -> bytes:
    values = values or {}
    abilities, technologies = skill_choices()
    choices = {
        "languages": LANGUAGES,
        "target_roles": ROLES,
        "skills": abilities,
        "technologies": technologies,
    }
    fields = {
        field: html.escape(str(values.get(field, "")), quote=True)
        for field in TEXT_FIELDS
    }
    for field, options in choices.items():
        fields[field] = checkboxes(field, options, list(values.get(field, [])))
    return PAGE.format(message=message, **fields).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(render())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = parse_qs(self.rfile.read(length).decode("utf-8"))
        form: dict[str, str | list[str]] = {
            field: raw.get(field, [""])[0] for field in TEXT_FIELDS
        }
        form.update({field: raw.get(field, []) for field in LIST_FIELDS})
        try:
            path = save_profile(form)
        except ValidationError as error:
            message = f'<div class="msg err">{html.escape(str(error))}</div>'
            self._send(render(form, message))
            return
        message = f'<div class="msg ok">Zapisano: {html.escape(path.name)}</div>'
        self._send(render(message=message))

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    server = HTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Formularz profilu: {url}  (Ctrl+C kończy)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
