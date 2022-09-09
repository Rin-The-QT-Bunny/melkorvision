from setuptools import setup, find_packages
setup(
    name="melkorvision",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="Melkor Logic Network.",

    # project main page
    url="http://jiayuanm.com/", 

    # the package that are prerequisites
    packages=find_packages(),
    package_data={
        '':['melkorvision'],
        'bandwidth_reporter':['melkorvision']
               },
)