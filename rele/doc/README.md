To build the documentation, invoke
```
	make html
```
Them point your browser to \_build/html/index.html.


#### External dependencies
To properly compile the documentation you must install

- [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
- [Sphinx](http://www.sphinx-doc.org)
- [Breathe](https://breathe.readthedocs.org)

In many unix distributions you can use the package manager to install them.

**Ubuntu**
```
apt-get install doxygen python-sphinx python-breathe
```
**Mac OS X**
```
brew install doxygen
pip install -U Sphinx
pip install breathe
```
