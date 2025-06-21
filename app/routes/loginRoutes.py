from flask import Blueprint
from app.controllers import loginController

main = Blueprint("main", __name__)

main.add_url_rule("/login", view_func=loginController.login_page, methods=["GET"])
main.add_url_rule("/login", view_func=loginController.login_submit, methods=["POST"])
main.add_url_rule("/logout", view_func=loginController.logout)