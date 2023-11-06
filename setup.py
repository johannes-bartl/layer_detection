from setuptools import setup


setup(name='layer_detection',
      python = "3.9",
      version='0.1',
      description='layer_detection',
      install_requires=[
            'tensorflow==2.14.0',
            # 'clickpoints==1.10.0',
            'numpy==1.25.2',
            'matplotlib==3.8.1',
            'tqdm',
            'opencv-python',
            'pyyaml',
            'tensorflow-addons',
            'imageio',
            'tifffile',
            'pyyaml'
      ])