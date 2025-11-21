from pathlib import Path
from dynaconf import Dynaconf, Validator

BASE_DIR = Path(__file__).parent.parent

Settings = Dynaconf(
    settings_files=[
        str(BASE_DIR / "config" / "defaults.toml"),
        str(BASE_DIR / "config.toml"),
    ],
    environments=True,
    envvar_prefix="GUIDES",
    load_dotenv=True,
    validators=[
        Validator("guides_dir", must_exist=True),
    ],
)

Settings.validators.validate()

CONF = {k.lower(): v for k, v in Settings.to_dict().items()}
