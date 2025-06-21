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
        flash("Selamat datang, {}! Anda berhasil login.".format(user.username), "success")  # Menampilkan username di flash message
        return redirect(url_for("admin.admin_dashboard"))
    else:
        flash("Username atau password yang Anda masukkan salah. Silakan coba lagi.", "danger")  # Menggunakan 'danger' untuk kesalahan
        return redirect(url_for("main.login_page"))

def logout():
    session.pop("user_id", None)
    flash("Anda telah berhasil logout.", "success")  # Flash logout
    return redirect(url_for("main.login_page"))
