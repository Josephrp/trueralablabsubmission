# Try the final Demo

in the command line run :

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/spaces/TeamTonic/TruEraMultiMed

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```
then:

```bash 
cd TruEraMultiMed
```
then :

```bash
pip install -r requirements.txt
```
Run with :

```bash
python app.py
```
Use by navidating your browser to :
```Localhost:3780```
