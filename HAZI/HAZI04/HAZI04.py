# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
def csv_to_df(test_data):
    df = pd.read_csv(test_data)
    return df

#df = csv_to_df("StudentsPerformance.csv")


# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def capitalize_columns(df):
    new_df = df.copy()
    for column in new_df:
        
        if not ('e' in column):
           new_df = new_df.rename(columns={column : column.upper()})

    return new_df

#print(capitalize_columns(df))



# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df):
    new_df = df.copy()
    passedcount = 0
    for index,row in new_df.iterrows():
        if row['math score'] >= 50:
            passedcount = passedcount+1

    return passedcount
#math_passed_count(df)

# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df):
    new_df = df.copy()
    subset = new_df[new_df['test preparation course'] == "completed"]
    return subset

#did_pre_course(df)

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%
def average_scores(df):
    new_df = df.copy()
    grouped = new_df.groupby(['parental level of education'])['math score','reading score','writing score'].mean()
    

    return grouped

#print(average_scores(df))
#average_scores(df)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%



def add_age(df):
    new_df = df.copy()
    new_df['age'] = ''
    np.random.seed(42)
    for i in range(len(new_df)):
        new_df['age'][i] = np.random.randint(low=18,high=67)


    return new_df

#print(add_age(csv_to_df('StudentsPerformance.csv')))

# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df):
    new_df = df.copy()
    best = 0
    for i in range(len(new_df)):
        if new_df['gender'][i] == 'female' and new_df['math score'][best] < new_df['math score'][i] and new_df['reading score'][best] < new_df['reading score'][i] and new_df['writing score'][best] < new_df['writing score'][i]:
            best = i



    return (new_df['math score'][best],new_df["reading score"][best],new_df["writing score"][best])

#female_top_score(df)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade(df):
    new_df = df.copy()
    new_df['grade'] = ''
    for i in range(len(new_df)):
        perc = ((new_df['math score'][i] + new_df['reading score'][i] + new_df['writing score'][i]) / 300)
        if perc < 0.6:
            new_df['grade'][i] = 'F'
        elif perc >= 0.6 and perc < 0.7:
            new_df['grade'][i] = 'D'
        elif perc >= 0.7 and perc < 0.8:
            new_df['grade'][i] = 'C'
        elif perc >= 0.8 and perc < 0.9:
            new_df['grade'][i] = 'B'
        elif perc >= 0.9:
            new_df['grade'][i] = 'A'




    return new_df

#add_grade(df)

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%
def math_bar_plot(df):
    new_df = df.copy()
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    grouped = new_df.groupby('gender')['math score'].mean()
    
    ax.bar(grouped.index,grouped.values)
    ax.set_ylabel('Math Score')
    ax.set_xlabel('Gender')
    ax.set_title('Average Math Score by Gender')

    return fig

#math_bar_plot(df)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df):
    new_df = df.copy()
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.hist(new_df['writing score'])
    ax.set_ylabel('Number of Students')
    ax.set_xlabel('Writing Score')
    ax.set_title('Distribution of Writing Scores')
    return fig

#writing_hist(df)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(df):
    new_df = df.copy()
    fig,ax = plt.subplots()
    grouped = new_df.groupby(['race/ethnicity'])['race/ethnicity'].count()
    ax.set_title('Proportion of Students by Race/Ethnicity')
    ax.pie(grouped,labels=grouped.index,autopct='%1.1f%%')
    
    return fig

#ethnicity_pie_chart(df)


