from teachDRL.teachers.algos.normalizing_teacher import NF
import numpy as np

### Test initialization
teacher = NF([0,1,2], [1,2,3])

### Test update

teacher.nb_random = 5
teacher.fit_rate = 5

task = np.array([1.,2., 3.], dtype=np.float32)
reward = 2.
for i in range(5):
    teacher.update(task, reward)

print(teacher.maf.sample(10))