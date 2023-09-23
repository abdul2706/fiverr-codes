import copy
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

    PLURALS = {'company': 'companies', 'building': 'buildings', 'department': 'departments', 'team': 'teams', 'raindrop': 'raindrops'}
    OUTPUT_FIELDS = {
        'company': ['item_type', 'item_id', 'item_name', 'admins', 'outbound_ip', 'roles', 'buildings'],
        'role': ['item_type', 'company_id', 'item_id', 'item_name', 'os_type'],
        'building': ['item_type', 'company_id', 'parent_id', 'item_id', 'item_name', 'admins', 'building_ip', 'departments'],
        'department': ['item_type', 'company_id', 'parent_id', 'item_id', 'item_name', 'admins', 'teams'],
        'team': ['item_type', 'company_id', 'building_id', 'parent_id', 'item_id', 'item_name', 'admins', 'raindrops'],
        'raindrop': ['item_type', 'company_id', 'building_id', 'department_id', 'item_id', 'parent_id', 'user', 'type', 'cpu_cores', 'RAM', 'disk_utilization', 'os', 'role', 'last_connected', 'last_location'],
    }

    input_data = json.load(open(input_json_path, 'r'))
    input_data = [obj for obj in input_data if 'item_type' in obj]
    output_obj = OrderedDict()
    for obj in input_data:
        item_type = obj['item_type']
        obj2 = {}
        for field in OUTPUT_FIELDS[item_type]:
            obj2[field] = obj.get(field, '')
            if field in ['roles', 'buildings', 'departments', 'teams', 'raindrops']:
                obj2[field] = []
        output_obj[f"{item_type}:{obj['item_id']}"] = obj2

    # add raindrops into their respective teams
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        if obj['item_type'] == 'raindrop':
            team_id = obj['parent_id']
            obj.pop('item_type')
            obj.pop('company_id')
            obj.pop('building_id')
            obj.pop('department_id')
            parent_key = f"team:{team_id}"
            if parent_key in output_obj:
                output_obj[parent_key]['raindrops'].append(obj)
            output_obj.pop(item_id)

    # add teams into their respective departments
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        if obj['item_type'] == 'team':
            department_id = obj['parent_id']
            obj.pop('item_type')
            obj.pop('company_id')
            obj.pop('building_id')
            obj.pop('parent_id')
            parent_key = f"department:{department_id}"
            if parent_key in output_obj:
                output_obj[parent_key]['teams'].append(obj)
            del output_obj[item_id]

    # add departments into their respective buildings
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        if obj['item_type'] == 'department':
            building_id = obj['parent_id']
            obj.pop('item_type')
            obj.pop('company_id')
            obj.pop('parent_id')
            parent_key = f"building:{building_id}"
            if parent_key in output_obj:
                output_obj[parent_key]['departments'].append(obj)
                del output_obj[item_id]

    # add buildings into their respective companies
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        if obj['item_type'] == 'building':
            company_id = obj['parent_id']
            obj.pop('item_type')
            obj.pop('company_id')
            obj.pop('parent_id')
            parent_key = f"company:{company_id}"
            if parent_key in output_obj:
                output_obj[parent_key]['buildings'].append(obj)
                del output_obj[item_id]

    # add roles into their respective companies
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        if obj['item_type'] == 'role':
            company_id = obj['company_id']
            obj.pop('item_type')
            obj.pop('company_id')
            parent_key = f"company:{company_id}"
            if parent_key in output_obj:
                output_obj[parent_key]['roles'].append(obj)
                del output_obj[item_id]

    # combine objects of same item_type into their own lists
    # e.g. list of companies, list of buildings, list of departments
    output_obj2 = copy.deepcopy(output_obj)
    for item_id, obj in output_obj2.items():
        item_type = PLURALS[item_id.split(':')[0]]
        if item_type not in output_obj:
            output_obj[item_type] = []
        obj.pop('item_type')
        output_obj[item_type].append(obj)
        output_obj.pop(item_id)

    # json_string = json.dumps(output_obj, indent=4)
    # print('json_string')
    # print(json_string)

    json.dump(output_obj, open(output_json_path, 'w'), indent=4)

    return output_obj

# input_json_path = 'sample_input.json'
input_json_path = 'sample_input_extended.json'
output_json_path = 'generated_output_bottom_up.json'
output_json = convert_json_to_hierarchy(input_json_path, output_json_path)
json_string = json.dumps(output_json, indent=4)
print('json_string')
print(json_string)
