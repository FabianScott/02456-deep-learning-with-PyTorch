import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

if __name__ == '__main__':
    notebook_filename = '4.2-EXE-CNN-CIFAR-10'
    with open(notebook_filename + '.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': ''}})
    with open(notebook_filename + 'executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
