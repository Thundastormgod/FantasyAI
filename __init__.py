from flask import Flask

def create_app():
    app = Flask(__name__)

    with app.app_context():
        # Import routes and models
        from . import app as app_routes

        # Register routes
        app.register_blueprint(app_routes)

    return app