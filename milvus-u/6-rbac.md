---
layout: default
title: 5. RBAC
nav_order: 5
description: ""
has_children: false
parent:  Milvus (U)
---

# Enabling RBAC in Milvus

Role-Based Access Control (RBAC) is a powerful feature in Milvus that allows you to manage access to specific resources, such as collections or partitions, by assigning permissions based on user roles and privileges. This feature is currently supported in Python and Java, and this guide will walk you through the process of enabling RBAC and managing users and roles within Milvus.

## Creating a User

To create a new user in Milvus, you can use the `create_user` method from the `pymilvus` utility module. For example:

```python
from pymilvus import utility

utility.create_user(user: str, password: str, using: str = "default") -> None
```

Here, `user` is the username for the new user, `password` is the password for this user, and `using` specifies the connection alias, which defaults to "default". 

Once a user is created, you have the ability to update the user’s password, list all users, and check the roles assigned to specific users. To update a user’s password, you would use the `update_password` method:

```python
utility.update_password(user: str, old_password: str, new_password: str, using: str = "default") -> None
```

In this method, you need to provide the username, the current password (`old_password`), the new password (`new_password`), and the connection alias.

To retrieve a list of all users, the `list_usernames` method can be used:

```python
utility.list_usernames(using: str = "default") -> List[str]
```

This will return a list of all usernames associated with the Milvus instance.

If you need to check the role information for a particular user, you can call the `list_user` method:

```python
utility.list_user(username: str, include_role_info: bool = False, using: str = "default") -> Dict
```

This method returns detailed information about the user, including their roles if the `include_role_info` parameter is set to `True`. Similarly, you can check roles for all users using the `list_users` method, which works in much the same way.

## Creating a Role

Creating roles is a central part of managing RBAC in Milvus. A role defines a set of privileges that can be assigned to users. To create a role, you can use the `Role` class provided by `pymilvus`. Here’s an example:

```python
from pymilvus import Role, utility

role_name = "roleA"
role = Role(role_name: str, using: str = "default")
role.create() -> None
```

In this example, `role_name` is the name of the role you want to create, and `using` specifies the connection alias.

After creating a role, you might want to check if the role exists, which can be done using the `is_exist` method:

```python
role.is_exist(role_name: str) -> bool
```

This method will return `True` if the role exists and `False` otherwise. You can also list all roles using the `list_roles` method:

```python
utility.list_roles(include_user_info: bool = False, using: str = "default") -> List[Dict]
```

This method will provide a list of all roles, optionally including the users associated with each role if `include_user_info` is set to `True`.

## Granting Privileges to a Role

Once a role is created, you can grant it specific privileges that define what actions the role can perform and on which resources. For example, to grant the permission to search all collections to a role, you can use the `grant` method:

```python
role.grant(object: str, object_name: str, privilege: str) -> None
```

Here, `object` refers to the type of resource (e.g., "Collection"), `object_name` can be the name of a specific resource or "*" for all resources, and `privilege` specifies the action (e.g., "Search").

If you are managing collections in multiple databases, ensure you are connected to the correct database before granting privileges. This can be done using `db.using_database()` or by directly connecting to the desired database.

After granting privileges, you can review them using methods like `list_grant` and `list_grants`. The `list_grant` method lists specific privileges granted to a role on a particular object:

```python
role.list_grant(object: str, object_name: str) -> List[Dict]
```

The `list_grants` method lists all privileges granted to the role:

```python
role.list_grants() -> List[Dict]
```

## Binding Roles to Users

To give a user the privileges associated with a role, you bind the role to the user. This is done with the `add_user` method:

```python
role.add_user(username: str) -> None
```

Here, `username` is the name of the user you want to bind to the role. Once a role is bound to a user, you can retrieve the list of users associated with the role using the `get_users` method:

```python
role.get_users() -> List[str]
```

This method returns a list of all users who have been assigned the role.

## Denying Access or Revoking Privileges

Denying access or revoking privileges should be done with caution, as these actions are irreversible. To remove a privilege from a role, use the `revoke` method:

```python
role.revoke(object: str, object_name: str, privilege: str) -> None
```

Similarly, to remove a user from a role, you would use the `remove_user` method:

```python
role.remove_user(username: str) -> None
```

If a role is no longer needed, it can be deleted using the `drop` method:

```python
role.drop(role_name: str) -> None
```

Finally, if you need to delete a user, you can do so with the `delete_user` method:

```python
utility.delete_user(user: str, using: str = "default") -> None
```

In all these methods, the parameters required are similar to those used during creation, granting, or binding operations.

