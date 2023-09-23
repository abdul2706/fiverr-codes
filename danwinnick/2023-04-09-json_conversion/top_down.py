from collections import OrderedDict
import json

def convert_json_to_hierarchy(input_json_path, output_json_path):
    # item_type hierarchy
    # -> company
    #   -> role
    #   -> building
    #     -> department
    #       -> team
    #         -> raindrop

    COMPANY_KEYS = ["item_id", "item_name", "admins", "outbound_ip", "roles", "buildings"]
    ROLE_KEYS = ["item_id", "item_name", "os_type"]
    BUILDING_KEYS = ["item_id", "item_name", "admins", "building_ip", "departments"]
    DEPARTMENT_KEYS = ["item_id", "item_name", "admins", "teams"]
    TEAM_KEYS = ["item_id", "item_name", "admins", "raindrops"]
    RAINDROP_KEYS = ["item_id", "parent_id", "user", "type", "cpu_cores", "RAM", "disk_utilization", "os", "role", "last_connected", "last_location"]

    input_data = json.load(open(input_json_path, 'r'))
    input_data = [obj for obj in input_data if 'item_type' in obj]
    print('[input_data]', len(input_data), type(input_data), type(input_data[0]))
    for i in range(len(input_data)):
        print(i, list(input_data[i].keys()))
    print()

    output_obj = OrderedDict()

    # extract all companies
    for obj in input_data:
        if obj['item_type'] == 'company':
            item_id = obj['item_id']
            company_obj = OrderedDict({key: obj.get(key, '') for key in COMPANY_KEYS})
            company_obj['roles'] = []
            company_obj['buildings'] = {}
            output_obj[item_id] = company_obj

    # extract all roles, and insert into respective companies
    for obj in input_data:
        if obj['item_type'] == 'role':
            company_id = obj['company_id']
            item_id = obj['item_id']
            role_obj = OrderedDict({key: obj.get(key, '') for key in ROLE_KEYS})
            output_obj[company_id]['roles'].append(role_obj)

    # extract all buildings, and insert into respective companies
    for obj in input_data:
        if obj['item_type'] == 'building':
            company_id = obj['company_id']
            item_id = obj['item_id']
            building_obj = OrderedDict({key: obj.get(key, '') for key in BUILDING_KEYS})
            building_obj['departments'] = {}
            output_obj[company_id]['buildings'][item_id] = building_obj

    # extract all departments, and insert into respective companies -> buildings
    for obj in input_data:
        if obj['item_type'] == 'department':
            company_id = obj['company_id']
            builing_id = obj['parent_id']
            item_id = obj['item_id']
            department_obj = OrderedDict({key: obj.get(key, '') for key in DEPARTMENT_KEYS})
            department_obj['teams'] = {}
            output_obj[company_id]['buildings'][builing_id]['departments'][item_id] = department_obj

    # extract all teams, and insert into respective companies -> buildings -> departments
    for obj in input_data:
        if obj['item_type'] == 'team':
            company_id = obj['company_id']
            builing_id = obj['building_id']
            department_id = obj['parent_id']
            item_id = obj['item_id']
            team_obj = OrderedDict({key: obj.get(key, '') for key in TEAM_KEYS})
            team_obj['raindrops'] = []
            output_obj[company_id]['buildings'][builing_id]['departments'][department_id]['teams'][item_id] = team_obj

    # extract all raindrops, and insert into respective companies -> buildings -> departments -> teams
    for obj in input_data:
        if obj['item_type'] == 'raindrop':
            company_id = obj['company_id']
            builing_id = obj['building_id']
            department_id = obj['department_id']
            team_id = obj['parent_id']
            item_id = obj['item_id']
            raindrop_obj = OrderedDict({key: obj.get(key, '') for key in RAINDROP_KEYS})
            output_obj[company_id]['buildings'][builing_id]['departments'][department_id]['teams'][team_id]['raindrops'].append(raindrop_obj)

    # convert output_obj to required output format
    final_output_obj = OrderedDict({"companies": []})
    for key, company in output_obj.items():
        buildings = list(company['buildings'].values())
        for building_idx in range(len(buildings)):
            departments = list(buildings[building_idx]['departments'].values())
            for department_idx in range(len(departments)):
                departments[department_idx]['teams'] = list(departments[department_idx]['teams'].values())
            buildings[building_idx]['departments'] = departments
        company['buildings'] = buildings
        final_output_obj['companies'].append(company)

    json.dump(final_output_obj, open(output_json_path, 'w'), indent=4)

    return final_output_obj

# input_json_path = 'sample_input.json'
input_json_path = 'sample_input_extended.json'
output_json_path = 'generated_output_top_down.json'
output_json = convert_json_to_hierarchy(input_json_path, output_json_path)
json_string = json.dumps(output_json, indent=4)
print('json_string')
print(json_string)
