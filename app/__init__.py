from flask import Flask
from config import SQLALCHEMY_DATABASE_URI
from app.extensions import db, migrate, socketio

def create_app():
    app = Flask(__name__)
    app.config.from_object("config")

    # Inisialisasi ekstensi
    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app)

    # Register semua blueprint
    from app.routes.loginRoutes import main      
    from app.routes.adminRoutes import admin      
    from app.routes.publicRoutes import public
    from app.routes.reviewRoutes import review  
    from app.routes.datasetRoutes import dataset  
    from app.routes.evaluationRoutes import evaluation  
    from app.routes.analyzeRoutes import analyze
    from app.routes.predictionRoutes import prediction     

    app.register_blueprint(main)    
    app.register_blueprint(admin)  
    app.register_blueprint(public)
    app.register_blueprint(review)
    app.register_blueprint(dataset)
    app.register_blueprint(evaluation)
    app.register_blueprint(analyze)
    app.register_blueprint(prediction)

    from app.models import userModel, reviewModel, evaluationModel, datasetModel

    return app
