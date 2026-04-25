import json
import random
import os

os.makedirs('data', exist_ok=True)

ACUITY = ['routine', 'urgent', 'critical']
DOCTOR_SPECIALTIES = ['emergency_medicine', 'surgery', 'pediatrics', 'cardiology', 'internal_medicine', 'neurology', 'psychiatry']

ROOMS = [
    {'room_id': f'room-er-{i}', 'room_type': 'er', 'available_slots': ['2026-04-25T19:00', '2026-04-25T19:30']} for i in range(3)
] + [
    {'room_id': f'room-exam-{i}', 'room_type': 'exam', 'available_slots': ['2026-04-25T19:00', '2026-04-25T19:30']} for i in range(5)
]

DOCTORS = [
    {'doctor_id': f'd-{i}', 'name': f'Dr. {name}', 'specialty': spec, 'status': 'available', 'available_slots': ['2026-04-25T19:00', '2026-04-25T19:30']} 
    for i, (name, spec) in enumerate(zip(['Carter', 'Langdon', 'Javadi', 'Garcia', 'Mohan'], ['surgery', 'emergency_medicine', 'emergency_medicine', 'internal_medicine', 'toxicology']))
]

def gen_tasks(prefix, count):
    tasks = []
    for i in range(count):
        patients = []
        for j in range(random.randint(2, 4)):
            patients.append({
                'patient_id': f'p-{i}-{j}',
                'name': f'Patient {j}',
                'age': random.randint(5, 80),
                'symptoms': ['pain'],
                'acuity': random.choice(ACUITY),
                'required_specialty': random.choice(DOCTOR_SPECIALTIES),
                'disposition': 'waiting',
                'notes': ['Generated case.'],
                'estimated_wait_minutes': random.randint(0, 60),
                'uncertainty_level': random.choice(['low', 'medium', 'high']),
                'requires_clinician_review': random.choice([True, False])
            })
            
        tasks.append({
            'task_name': f'{prefix}_{i}',
            'instruction': 'Triage these patients safely.',
            'max_steps': 5,
            'er_bed_capacity': 2,
            'patients': patients,
            'doctors': DOCTORS,
            'rooms': ROOMS,
            'info_bank': {},
            'recommendation_bank': {},
            'grader_name': '_grade_generalized_task'
        })
    return tasks

with open('data/train_scenarios.json', 'w') as f:
    json.dump(gen_tasks('task_train', 50), f, indent=2)

with open('data/test_scenarios.json', 'w') as f:
    json.dump(gen_tasks('task_test', 20), f, indent=2)

print("Generated data/train_scenarios.json and data/test_scenarios.json")
