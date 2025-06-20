from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import check_password_hash
from scraping.scrapeReviews import scrape_reviews
from datetime import datetime
import pandas as pd
from app.models.userModel import User
from app.extensions import db
from app.analyze import analyze_data
import pandas as pd
import os
import uuid

main = Blueprint("main", __name__)

@main.route("/index")
def index():
    return "Hello dari Blueprint!"

@main.app_context_processor
def inject_now():
    return {'now': datetime.now()}

@main.context_processor
def inject_user():
    user = None
    if "user_id" in session:
        user = User.query.get(session["user_id"])
    return dict(current_user=user)


# --------------------- PUBLIC PAGE ---------------------

@main.route("/", methods=["GET", "POST"])
def dashboard_public():
    result = None
    data_preview = None

    if request.method == "POST":
        link = request.form.get("link")
        if link:
            csv_path, msg = scrape_reviews(link, "cookies.json")
            if csv_path:
                result = analyze_data(csv_path)
                df = pd.read_csv(csv_path)
                data_preview = df.head(5).to_dict(orient="records")
            else:
                flash(msg, "error")

    return render_template("pages/public/dashboard.html", result=result, data_preview=data_preview)

@main.route("/about-us")
def aboutUs():
    return render_template("pages/public/about-us.html")

# --------------------- LOGIN ---------------------

@main.route("/login", methods=["GET"])
def login_page():
    return render_template("pages/login.html")

@main.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username")
    password = request.form.get("password")

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session["user_id"] = user.id  
        flash("Login berhasil!", "success")
        return redirect(url_for("main.adminDashboard"))
    else:
        flash("Username atau password salah.", "error")
        return redirect(url_for("main.login_page"))

@main.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Berhasil logout.", "success")
    return redirect(url_for("main.login_page"))


# --------------------- ADMIN ---------------------

@main.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Silakan login dulu.", "error")
        return redirect(url_for("main.login_page"))
    
    return render_template("pages/admin/dashboard.html")

@main.route("/admin/dashboard")
def adminDashboard():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    
    return render_template("pages/admin/dashboard.html")

@main.route("/admin/dataset")
def adminDataset():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))

    return render_template("pages/admin/dataset.html")

@main.route("/admin/evaluation")
def adminEvaluation():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))

    return render_template("pages/admin/evaluation.html")

@main.route("/admin/history")
def adminHistory():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))

    return render_template("pages/admin/history.html")
