from flask import render_template, request, redirect, url_for, flash, session
from app.models.userModel import User

def login_page():
    return render_template("pages/login.html")

def login_submit():
    username = request.form.get("username")
    password = request.form.get("password")

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session["user_id"] = user.id
        flash("<strong>Berhasil Login!</strong><br>Selamat datang, <strong>{}</strong>.".format(user.username), "success")
        return redirect(url_for("admin.admin_dashboard"))
    else:
        flash("<strong>Login Gagal!</strong><br>Username atau password salah.", "error")
        return redirect(url_for("main.login_page"))

def logout():
    session.pop("user_id", None)
    flash("👋 <strong>Logout berhasil!</strong><br>Sampai jumpa lagi!", "success")
    return redirect(url_for("main.login_page"))
