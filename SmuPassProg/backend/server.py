from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
USERS_CSV = BASE_DIR / "users.csv"
POP_PATH = BASE_DIR / "population_scores.csv"

app = Flask(__name__, static_folder="web", static_url_path="")

# 세션 키 (실서비스면 환경변수로 분리)
app.secret_key = "dev-secret-change-this"


def hash_password(password: str) -> str:
    """간단한 SHA256 해시 (프로젝트용)."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_users_csv() -> None:
    """users.csv 파일이 없으면 헤더 포함해서 생성."""
    if not USERS_CSV.exists():
        with USERS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "username",
                    "password_hash",
                    "name",
                    "role",
                    "school",
                    "grade",
                    "college",
                    "email",
                ]
            )


def load_users() -> list[dict]:
    """users.csv 전체를 리스트(dict)로 로딩."""
    ensure_users_csv()
    with USERS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_user(username: str) -> dict | None:
    """username으로 사용자 한 명 찾기."""
    for row in load_users():
        if row.get("username") == username:
            return row
    return None


def add_user(
    username: str,
    password: str,
    name: str,
    role: str = "",
    school: str = "",
    grade: str = "",
    college: str = "",
    email: str = "",
) -> None:
    """새로운 사용자 한 명을 users.csv에 추가."""
    ensure_users_csv()
    with USERS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                username,
                hash_password(password),
                name,
                role,
                school,
                grade,
                college,
                email,
            ]
        )


def load_population_scores():
    """population_scores.csv 로딩 (지금은 통계용 예시)."""
    if not POP_PATH.exists():
        print(">>> population_scores.csv 파일을 찾을 수 없습니다.")
        return None, None
    print(">>> population_scores.csv 로딩 중...")
    df = pd.read_csv(POP_PATH)
    if "total_score" in df.columns:
        mean_cutoff = df["total_score"].mean()
        median_cutoff = df["total_score"].median()
        trimmed_cutoff = mean_cutoff  # 필요 시 절사 평균으로 교체
        cutoff_info = {
            "mean_cutoff": float(mean_cutoff),
            "median_cutoff": float(median_cutoff),
            "trimmed_cutoff": float(trimmed_cutoff),
        }
    else:
        cutoff_info = None
    print(">>> cutoff_info:", cutoff_info)
    return df, cutoff_info


population_df, cutoff_info = load_population_scores()


# ------------- 라우트 -------------


@app.route("/")
def index():
    """
    루트: 항상 로그인 페이지를 보여준다.
    (세션이 있어도 자동으로 selectPage로 보내지 않음)
    """
    return app.send_static_file("loginPage.html")


@app.route("/loginPage.html")
def login_page_direct():
    """직접 loginPage.html로 접근해도 동일하게 동작."""
    return app.send_static_file("loginPage.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """회원가입: GET은 폼, POST는 CSV에 저장."""
    if request.method == "GET":
        return app.send_static_file("signup.html")

    username = (request.form.get("username") or "").strip()
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "").strip()
    school = (request.form.get("school") or "").strip()
    grade = (request.form.get("grade") or "").strip()
    college = (request.form.get("college") or "").strip()
    email = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    password_confirm = request.form.get("passwordConfirm") or ""

    # 간단 검증
    if not username or not name or not email or not password:
        return redirect(url_for("signup"))

    if password != password_confirm:
        return redirect(url_for("signup"))

    if find_user(username) is not None:
        # 이미 존재하는 아이디
        return redirect(url_for("signup"))

    add_user(
        username=username,
        password=password,
        name=name,
        role=role,
        school=school,
        grade=grade,
        college=college,
        email=email,
    )

    # 회원가입 후 로그인 페이지로 이동
    return redirect(url_for("index"))


@app.route("/login", methods=["POST"])
def login():
    """
    로그인: users.csv를 읽어서 검증.
    실패하면 /?error=1 로 보내서 로그인 페이지에서 팝업 띄우기.
    """
    username = (request.form.get("id") or request.form.get("username") or "").strip()
    password = request.form.get("pw") or request.form.get("password") or ""

    # 아이디/비번 미입력
    if not username or not password:
        return redirect(url_for("index", error="1"))

    user = find_user(username)
    if user is None:
        # 존재하지 않는 아이디
        return redirect(url_for("index", error="1"))

    if user.get("password_hash") != hash_password(password):
        # 비밀번호 불일치
        return redirect(url_for("index", error="1"))

    # 로그인 성공
    session["user"] = username
    return redirect(url_for("select_page"))


@app.route("/logout")
def logout():
    """로그아웃."""
    session.clear()
    return redirect(url_for("index"))


@app.route("/selectPage.html")
def select_page():
    """선택 페이지: 로그인한 사람만 접근 가능."""
    if "user" not in session:
        return redirect(url_for("index", error="1"))
    return app.send_static_file("selectPage.html")


@app.route("/report.html")
def report_page():
    """리포트 페이지: 로그인한 사람만 접근 가능."""
    if "user" not in session:
        return redirect(url_for("index", error="1"))
    return app.send_static_file("report.html")


@app.route("/api/report", methods=["GET"])
def api_report():
    """나중에 프론트에서 백엔드 데이터를 가져오고 싶을 때 사용."""
    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401

    data = {
        "items": [
            {"id": "a1", "name": "학업성취도", "desc": "학업역량", "score": 91},
            {"id": "a2", "name": "학업태도", "desc": "학업역량", "score": 84},
            {"id": "a3", "name": "탐구력", "desc": "학업역량", "score": 87},
            {"id": "c1", "name": "전공(계열) 관련 교과 이수 노력", "desc": "진로역량", "score": 78},
            {"id": "c2", "name": "전공(계열) 관련 교과 성취도", "desc": "진로역량", "score": 82},
            {"id": "c3", "name": "진로 탐색 활동과 경험", "desc": "진로역량", "score": 73},
            {"id": "s1", "name": "협업과 소통능력", "desc": "공동체역량", "score": 88},
            {"id": "s2", "name": "나눔과 배려", "desc": "공동체역량", "score": 75},
            {"id": "s3", "name": "성실성과 규칙준수", "desc": "공동체역량", "score": 93},
            {"id": "s4", "name": "리더십", "desc": "공동체역량", "score": 80},
        ],
        "cutline": 850,
        "rank": 28,
        "totalCandidate": 97,
    }
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
