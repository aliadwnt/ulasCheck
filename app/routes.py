# from flask import Flask, render_template

# app = Flask(__name__, template_folder="templates")

# @app.route("/")
# def dashboard_public():
#     return render_template("pages/public/dasboard.html")
from flask import Flask, render_template

app = Flask(__name__, template_folder="templates")

@app.route("/")
def dashboard_public():
    return render_template("pages/public/dashboard.html")

@app.route("/about-us")
def aboutUs():
    return render_template("pages/public/about-us.html")

@app.route("/login")
def login():
    return render_template("pages/login.html")

@app.route("/admin/dashboard")
def adminDashboard():
    return render_template("pages/admin/dashboard.html")

@app.route("/admin/dataset")
def adminDataset():
    return render_template("pages/admin/dataset.html")

@app.route("/admin/evaluation")
def adminEvaluation():
    return render_template("pages/admin/evaluation.html")

@app.route("/admin/hitsory")
def adminHistory():
    return render_template("pages/admin/history.html")

# @app.route("/admin/evaluasi")
# def admin_evaluasi():
#     return render_template("pages/admin/evaluasi.html")

if __name__ == "__main__":
    app.run(debug=True)
