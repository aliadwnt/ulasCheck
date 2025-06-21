from flask import Blueprint
from app.controllers import adminController

admin = Blueprint("admin", __name__)

admin.add_url_rule("/dashboard", view_func=adminController.dashboard)
admin.add_url_rule("/admin/dashboard", view_func=adminController.admin_dashboard)
admin.add_url_rule("/admin/dataset", view_func=adminController.admin_dataset)
admin.add_url_rule("/admin/evaluation", view_func=adminController.admin_evaluation)
admin.add_url_rule("/admin/history", view_func=adminController.admin_history)