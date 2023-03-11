import json

def isValid(stale, latest, otjson):
    cursor_pos = 0
    ot_list = json.loads(otjson)

    # if ot_list is empty
    if not ot_list:
        return stale == latest

    transformation_str = stale
    for operation in ot_list:
        operation_type = operation["op"]

        # insert operation
        if operation_type == "insert":
            insertion_str = operation["chars"]
            transformation_str = transformation_str[:cursor_pos] + insertion_str + transformation_str[cursor_pos:]
            cursor_pos += len(insertion_str)

        # delete operation
        if operation_type == "delete":
            delete_count = operation["count"]
            if cursor_pos + delete_count > len(transformation_str):
                return False
            transformation_str = transformation_str[0: cursor_pos] + transformation_str[cursor_pos+delete_count:]

        # skip operation
        if operation_type == "skip":
            cursor_pos += operation["count"]
            if cursor_pos > len(transformation_str):
                return False
            
    return transformation_str == latest

print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'Repl.it uses operational transformations.',
  '[{"op": "skip", "count": 40}, {"op": "delete", "count": 47}]'
))

print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'Repl.it uses operational transformations.',
  '[{"op": "skip", "count": 45}, {"op": "delete", "count": 47}]'
))

print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'Repl.it uses operational transformations.',
  '[{"op": "skip", "count": 40}, {"op": "delete", "count": 47}, {"op": "skip", "count": 2}]'
))

print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'We use operational transformations to keep everyone in a multiplayer repl in sync.',
  '[{"op": "delete", "count": 7}, {"op": "insert", "chars": "We"}, {"op": "skip", "count": 4}, {"op": "delete", "count": 1}]'
))
  
print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'We can use operational transformations to keep everyone in a multiplayer repl in sync.',
  '[{"op": "delete", "count": 7}, {"op": "insert", "chars": "We"}, {"op": "skip", "count": 4}, {"op": "delete", "count": 1}]'
))

print(isValid(
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  'Repl.it uses operational transformations to keep everyone in a multiplayer repl in sync.',
  '[]'
))