from freezegun import freeze_time
from reichlab_python_template.app import main


@freeze_time("2019-07-13")
def test_main_date():
    output = main()
    assert "July 13, 2019" in output
