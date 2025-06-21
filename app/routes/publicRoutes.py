from flask import Blueprint, request
from app.controllers import publicController

public = Blueprint("public", __name__)

@public.context_processor
def inject_user_and_now():
    return publicController.inject_user_and_now()

public.add_url_rule("/", view_func=publicController.index, methods=["GET", "POST"])
public.add_url_rule("/review/<int:id>", view_func=publicController.show_review)
public.add_url_rule("/download/<int:id>", view_func=publicController.download_file, methods=["POST"])
public.add_url_rule("/analyze/<int:id>", view_func=publicController.analyze_file, methods=["POST"])
public.add_url_rule("/cancel/<int:id>", view_func=publicController.cancel)
public.add_url_rule("/about-us", view_func=publicController.about_us)