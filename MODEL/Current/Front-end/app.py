from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import requests
import numpy as np

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# PostgreSQL connection URI format:
# postgresql://username:password@host:port/databasename
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://fsl_admin:admin123@localhost:5432/fsl_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24)  # Required for session management

app.config['DEV_AUTO_LOGIN'] = True

if not app.config['DEV_AUTO_LOGIN']:
    db = SQLAlchemy(app)
else:
    db = None



if db:
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        password = db.Column(db.String(120), nullable=False)
        progress = db.Column(db.Float, default=0.0)



@app.before_request
def _dev_auto_login():
    """Automatically log in when running locally with debug=True and DEV_AUTO_LOGIN=True."""
    if not app.config.get("DEV_AUTO_LOGIN"):
        return
    if not app.debug:
        return
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return
    if not session.get("authenticated"):
        session["authenticated"] = True
        session["user"] = "dev"



def register_user(username, password):
    if not db:
        return False, "Database is disabled in DEV mode"
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return False, "Username already exists"
    new_user = User(username=username, progress=0.0)
    new_user.password = password
    try:
        db.session.add(new_user)
        db.session.commit()
        return True, "Registration successful"
    except Exception as e:
        db.session.rollback()
        return False, f"Registration failed: {str(e)}"


def authenticate_user(username, password):
    # Quick admin backdoor
    if username == "admin" and password == "fsl2025":
        return True
    if not db:
        return False
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        return True
    return False



def login_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get("authenticated"):
            flash("Please login to access this page", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/tutor")
@login_required
def tutor():
    return render_template("tutor.html")


@app.route("/learn/<category>")
@login_required
def learn_category(category):
    valid_categories = ['numbers', 'colors', 'shapes', 'family']
    if category not in valid_categories:
        flash("Invalid category", "error")
        return redirect(url_for("tutor"))
    return render_template("tutor.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if authenticate_user(username, password):
            session["user"] = username
            session["authenticated"] = True
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if not all([username, password, confirm_password]):
            flash("All fields are required", "error")
        elif password != confirm_password:
            flash("Passwords do not match", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters long", "error")
        else:
            success, message = register_user(username, password)
            if success:
                flash("Registration successful! Please login.", "success")
                return redirect(url_for("login"))
            else:
                flash(message, "error")
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "info")
    return redirect(url_for("home"))


@app.route("/activity")
@login_required
def activity():
    return render_template("activity.html")


@app.route("/results")
@login_required
def results():
    return render_template("results.html")



@app.route("/api/phrases")
def api_phrases():
    phrases = ["father", "mother", "red", "blue", "one", "two"]
    return jsonify({"phrases": phrases})


@app.route("/api/random", methods=["GET"])
def api_random():
    import random
    phrases = ["father", "mother", "red", "blue", "one", "two"]
    return jsonify({"phrase": random.choice(phrases)})


@app.route("/api/teach", methods=["GET"])
def api_teach():
    phrase = (request.args.get("phrase") or "").strip().lower()
    sample = {
        "father": {"video": "/static/video/family_father.mp4", "steps": ["Touch forehead", "Move down"]},
        "mother": {"video": "/static/video/family_mother.mp4", "steps": ["Touch chin", "Move down"]}
    }
    return jsonify(sample.get(phrase, {"video": "", "steps": []}))


# === MAIN API CONNECTION TO AI MODEL ===
@app.route("/api/assess", methods=["POST"])
def api_assess():
    data = request.get_json(force=True) or {}
    target = (data.get("target") or "").strip()
    frames = data.get("frames") or []
    features = data.get("features") or None

    # Step 1: Handle missing features (temporary random features)
    if not features:
        features = np.random.rand(188).tolist()

    # Step 2: Send to backend model
    try:
        res = requests.post("http://127.0.0.1:5000/predict", json={"features": features})
        res.raise_for_status()
        pred = res.json().get("prediction", "unknown")
    except Exception as e:
        return jsonify({"error": f"Model backend not reachable: {str(e)}"}), 500

    # Step 3: Compare prediction vs. target
    correct = pred.lower() == target.lower()

    # Step 4: Respond to frontend
    return jsonify({
        "prediction": pred,
        "target": target,
        "correct": correct
    })


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


# === RUN APP ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
