import os

class Config(object):
    """Parent configuration class."""
    DEBUG = False
    CSRF_ENABLED = True
    #SECRET = os.getenv('SECRET')
    VIDEO_CATALOG = ['0', '2', '4']
    # VIDEO_FILES_PATH = 'static/videos'
    SUPPORTED_QUALITIES = list(range(1, 6))
    ENABLE_PREFETCHING = os.getenv('ENABLE_PREFETCHING') == 'true'
    T_VERT = 4
    T_HOR = 4
class DevelopmentConfig(Config):
    """Configurations for Development."""
    DEBUG = True


class TestingConfig(Config):
    """Configurations for Testing, with a separate test database."""
    TESTING = True
    DEBUG = True

class StagingConfig(Config):
    """Configurations for Staging."""
    DEBUG = True


class ProductionConfig(Config):
    """Configurations for Production."""
    DEBUG = False
    TESTING = False


app_config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'staging': StagingConfig,
    'production': ProductionConfig
}
