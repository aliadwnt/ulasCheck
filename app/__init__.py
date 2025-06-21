from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
from config import SQLALCHEMY_DATABASE_URI

# Inisialisasi ekstensi
db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO(cors_allowed_origins="*")

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

    app.register_blueprint(main)    
    app.register_blueprint(admin)  
    app.register_blueprint(public)  

    # Import semua model agar dikenali oleh migrasi
    from app.models import userModel, reviewModel, predictionModel, evaluationModel, datasetModel

    return app
