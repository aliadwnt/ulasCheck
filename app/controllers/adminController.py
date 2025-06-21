from flask import render_template, session, redirect, url_for, flash

def dashboard():
    if "user_id" not in session:
        flash("Silakan login dulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dashboard.html")

def admin_dashboard():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dashboard.html")

def admin_dataset():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/dataset.html")

def admin_evaluation():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/evaluation.html")

def admin_history():
    if "user_id" not in session:
        flash("Silakan login terlebih dahulu.", "error")
        return redirect(url_for("main.login_page"))
    return render_template("pages/admin/history.html")
