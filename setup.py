from distutils.core import setup

setup(
    name='kern-smooth',
    version='1.0.0',
    packages=['kern_smooth'],
    install_requires=['numpy', 'pandas', 'scipy'],
    url='https://github.com/AlexanderButyaev/kern_smooth',
    license='MIT',
    author='Alexander Butyaev',
    author_email='alexander.butyaev@mail.mcgill.ca',
    description='A python implementation of KernSmooth package (https://cran.r-project.org/web/packages/KernSmooth):' + \
     			'kernel smoothing and density estimation functions based on the book: '+ \
     			'Wand, M.P. and Jones, M.C. (1995) "Kernel Smoothing".'
)