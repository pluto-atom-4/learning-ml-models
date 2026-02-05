import matplotlib.pyplot as plt
import seaborn as sns


def plot_urgency_by_age(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df[df['Urgency'] == 1], x='age', bins=20, kde=True, color='red')
    plt.title("Age Group with Most Urgent Need (Urgency = 1)")
    plt.show()


def plot_common_symptoms_urgent(df):
    symptoms = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
    urgent_patients = df[df['Urgency'] == 1][symptoms].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    urgent_patients.plot(kind='bar', color='orange')
    plt.title("Most Common Symptoms for Urgent Patients")
    plt.ylabel("Frequency")
    plt.show()


def compare_cough_by_urgency(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Urgency', y='cough', estimator=lambda x: sum(x) / len(x))
    plt.title("Cough Prevalence: No Urgency (0) vs Urgent (1)")
    plt.ylabel("Proportion with Cough")
    plt.show()
