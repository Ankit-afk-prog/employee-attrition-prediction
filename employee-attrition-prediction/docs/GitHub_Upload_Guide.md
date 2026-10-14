# GitHub Upload Guide

Step-by-step instructions for publishing this project as the GitHub repository `employee-attrition-prediction`.

> Nothing has been uploaded yet. These steps are for you to carry out.

## 1. Check the folder before uploading

The repository root should contain:

```
README.md   requirements.txt   .gitignore
data/IBM_HR_Analytics_Employee_Attrition.csv   data/README.md
notebooks/Employee_Attrition_Prediction.ipynb
src/attrition_features.py   src/predict.py
models/final_model.joblib   visualizations/*.png   results/*.csv
docs/Project_Report.md   docs/Presentation_Content.md   docs/GitHub_Upload_Guide.md
```

The `.venv/` folder (virtual environment) must **not** be uploaded. `.gitignore` already excludes it.

**Dataset licence.** The IBM HR dataset is published on Kaggle under ODbL v1.0 / DbCL v1.0, which permit redistribution with attribution. The attribution is in `README.md`. If your college prefers that you do not redistribute data, delete `data/IBM_HR_Analytics_Employee_Attrition.csv` before committing. The README already explains how to download the file from Kaggle.

## 2. Create the repository on GitHub

1. Sign in at <https://github.com>.
2. Click **+** (top right) → **New repository**.
3. Repository name: `employee-attrition-prediction`.
4. Description: *Employee attrition prediction with Logistic Regression, Random Forest and XGBoost – Predictive Analytics CA3*.
5. Choose **Public** (or **Private** if your faculty requires it).
6. **Do not** tick "Add a README", ".gitignore" or "license". This project already has them, and an empty repository avoids merge conflicts.
7. Click **Create repository** and copy its URL, e.g. `https://github.com/Ankit-afk-prog/employee-attrition-prediction.git`.

## 3a. Upload with Git (recommended)

Open a terminal in the project folder (adjust the path to where the folder is on your computer):

```bash
cd path/to/employee-attrition-prediction
```

Initialise a repository and make the first commit:

```bash
git init -b main
```

```bash
git add .
```

```bash
git status
```

Check that `.venv/` is **not** listed, then commit:

```bash
git commit -m "Employee attrition prediction: notebook, report, presentation, model"
```

Connect to GitHub and push (replace `Ankit-afk-prog`):

```bash
git remote add origin https://github.com/Ankit-afk-prog/employee-attrition-prediction.git
```

```bash
git push -u origin main
```

If Git asks for a password, use a GitHub **personal access token** (GitHub → Settings → Developer settings → Personal access tokens) or sign in with the GitHub CLI (`gh auth login`). Your account password will not work.

## 3b. Upload through the GitHub website (no Git needed)

1. On the new empty repository page, click **uploading an existing file**.
2. Drag in the files from the project root: `README.md`, `requirements.txt` and `.gitignore`.
3. To keep folders, drag the **folders** `data`, `notebooks`, `src`, `models`, `visualizations`, `results` and `docs` themselves (supported in Chrome and Edge). Otherwise use **Add file → Create new file**, type `data/` to create the folder, then upload into it.
4. Write a commit message and click **Commit changes**.
5. Do **not** upload `.venv/`.

Note: `.gitignore` is a hidden file on macOS. Press `Cmd + Shift + .` in Finder to show it.

## 4. Check that everything renders

1. Open `notebooks/Employee_Attrition_Prediction.ipynb` on GitHub. It should display with all outputs and charts, because it was saved after execution. If GitHub shows "Unable to render", reload the page, or view it at `https://nbviewer.org/github/Ankit-afk-prog/employee-attrition-prediction/blob/main/notebooks/Employee_Attrition_Prediction.ipynb`.
2. Open `README.md` (shown on the repository front page) and `docs/Project_Report.md`. Tables and the embedded figures should display, and the equations should render.
3. Click a few figure links to confirm that `visualizations/` was uploaded.

## 5. Updating after changes

If you edit the notebook or re-run it, re-run it top to bottom first:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Employee_Attrition_Prediction.ipynb
```

Update the README, report and slides if any number changed. The notebook writes the source tables to `results/`. Then commit and push:

```bash
git add .
```

```bash
git commit -m "Update results"
```

```bash
git push
```

On the website, use **Add file → Upload files** and upload the changed files. Files with the same name are replaced.

## 6. Final checklist

- [ ] Placeholders (`[STUDENT NAME]` etc.) in `Project_Report.md` and `Presentation_Content.md` filled in
- [ ] Notebook shows outputs on GitHub
- [ ] README results match the notebook
- [ ] `.venv/` not uploaded
- [ ] Dataset either included with attribution, or removed with download instructions
- [ ] Repository link added to your report or submission form, if required
