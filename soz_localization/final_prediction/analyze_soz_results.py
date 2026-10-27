import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl



# ==============================
# NPJ STYLE
# ==============================

mpl.rcParams.update({

    "font.family":"Times New Roman",

    "font.size":8,

    "pdf.fonttype":42,

    "ps.fonttype":42

})


RESULT_DIR = (
"downstream/soz_localization/final_prediction"
)


FIG_DIR = (
"downstream/soz_localization/final_prediction/figures"
)


os.makedirs(
    FIG_DIR,
    exist_ok=True
)



files = sorted(
glob.glob(
RESULT_DIR+
"/*_SOZ_probability.csv"
)
)



records=[]



for f in files:

    df=pd.read_csv(f)


    patient=df.patient.iloc[0]


    prob=df.SOZ_probability.values

    label=df.clinical_SOZ.values



    order=np.argsort(
        prob[::-1]
    )


    ranked=np.argsort(
        -prob
    )



    for k in [5,10,20]:

        top=ranked[:k]


        hit=(
            label[top].sum()>0
        )


        records.append({

            "patient":patient,

            "TopK":k,

            "Hit":int(hit)

        })



result=pd.DataFrame(records)



print(result)



# ==============================
# Top K Accuracy Figure
# ==============================


summary=result.groupby(
"TopK"
).Hit.mean()



plt.figure(
figsize=(3,2)
)


plt.plot(
summary.index,
summary.values,
marker="o"
)


plt.xlabel(
"Number of top-ranked contacts (K)"
)


plt.ylabel(
"Localization hit rate"
)


plt.ylim(
0,
1
)


plt.tight_layout()


plt.savefig(
os.path.join(
FIG_DIR,
"Figure_TopK_localization.pdf"
),
dpi=600
)


plt.savefig(
os.path.join(
FIG_DIR,
"Figure_TopK_localization.png"
),
dpi=600
)



print(
"Saved Figure_TopK_localization"
)



# ==============================
# Patient heatmap
# ==============================


matrix=[]


patients=[]


for f in files:

    df=pd.read_csv(f)


    p=df.patient.iloc[0]


    values=np.sort(
        df.SOZ_probability.values
    )[::-1]


    matrix.append(values)

    patients.append(p)



maxlen=max(
len(x)
for x in matrix
)



heat=np.zeros(
(maxlen,len(matrix))
)


for i,x in enumerate(matrix):

    heat[:len(x),i]=x



plt.figure(
figsize=(6,5)
)


plt.imshow(
heat,
aspect="auto"
)


plt.colorbar(
label="SOZ probability"
)


plt.xticks(
range(len(patients)),
patients,
rotation=90,
fontsize=5
)


plt.ylabel(
"Ranked contacts"
)


plt.tight_layout()


plt.savefig(
os.path.join(
FIG_DIR,
"Figure_SOZ_probability_heatmap.pdf"
),
dpi=600
)


plt.savefig(
os.path.join(
FIG_DIR,
"Figure_SOZ_probability_heatmap.png"
),
dpi=600
)


print(
"Saved heatmap"
)
