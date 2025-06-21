from flask import Blueprint, render_template, session, redirect, url_for, flash

admin = Blueprint("admin", __name__)

@admin.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Silakan login dulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dashboard.html")

@admin.route("/admin/dashboard")
def adminDashboard():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dashboard.html")

@admin.route("/admin/dataset")
def adminDataset():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dataset.html")

@admin.route("/admin/evaluation")
def adminEvaluation():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/evaluation.html")

@admin.route("/admin/history")
def adminHistory():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/history.html")
