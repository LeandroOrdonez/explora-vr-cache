import os

from app import create_app

config_name = os.getenv('APP_SETTINGS') # config_name = "development"
app = create_app(config_name)
#app.app_context().push() # prevent 'No application found. Either work inside a view function or push an application context.': https://flask-sqlalchemy.palletsprojects.com/en/2.x/contexts/

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
