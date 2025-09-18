import pandas as pd
import random
from faker import Faker

fake = Faker()

departments = ["HR", "IT", "Finance", "Marketing", "Sales", "Operations"]
genders = ["M", "F"]

data = []

for i in range(100):  # generate 100 rows
    employee_id = 1000 + i
    name = fake.first_name()
    age = random.randint(22, 60)
    gender = random.choice(genders)
    department = random.choice(departments)
    join_date = fake.date_between(start_date='-10y', end_date='today')
    salary = random.randint(40000, 120000)
    performance_score = round(random.uniform(3.0, 5.0), 1)
    
    data.append([employee_id, name, age, gender, department, join_date, salary, performance_score])

df = pd.DataFrame(data, columns=["EmployeeID", "Name", "Age", "Gender", "Department", "JoinDate", "Salary", "PerformanceScore"])

df.to_csv("professional_dataset.csv", index=False)
print("Professional dataset created!")
