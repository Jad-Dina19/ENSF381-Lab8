import { useEffect, useState } from 'react';
import { API_BASE_URL, readJson } from '../api';
import AddUserForm from './AddUserForm';
import Controls from './Controls';
import UserList from './UserList';

const API_URL = `${API_BASE_URL}/users`;

function sortUsers(users, sortBy) {
  return [...users].sort((left, right) => {
    if (sortBy === 'group') {
      return Number(left.user_group) - Number(right.user_group);
    }

    return Number(left.id) - Number(right.id);
  });
}

function UserDirectoryPage() {
  const [users, setUsers] = useState([]);
  const [sortBy, setSortBy] = useState('id');
  const [viewMode, setViewMode] = useState('grid');
  const [feedback, setFeedback] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  async function loadUsers(activeSortBy = sortBy) {
    setIsLoading(true);

    try {
      const response = await fetch(API_URL);
      const data = await readJson(response);
      if (!response.ok) {
        throw new Error(data.message || `Fetch failed: HTTP ${response.status}`);
      }

      setUsers(sortUsers(data, activeSortBy));
      setFeedback(null);
    } catch (error) {
      setUsers([]);
      setFeedback({
        type: 'error',
        text: error.message,
      });
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    loadUsers('id');
  }, []);

  function sortUsersByGroup() {
    setUsers((currentUsers) => sortUsers(currentUsers, 'group'));
    setSortBy('group');
  }

  function sortUsersById() {
    setUsers((currentUsers) => sortUsers(currentUsers, 'id'));
    setSortBy('id');
  }

  function toggleViewMode() {
    setViewMode((currentViewMode) => (currentViewMode === 'grid' ? 'list' : 'grid'));
  }

  async function deleteUser(userId) {
    const idToDelete = userId.trim();
    if (!idToDelete) {
      throw new Error('Enter a user ID before trying to delete.');
    }

    try {
      const response = await fetch(`${API_URL}/${idToDelete}`, {
        method: 'DELETE',
        headers: {
          Accept: 'application/json',
        },
      });
      const data = await readJson(response);

      if (!response.ok) {
        throw new Error(data.message || `Delete failed: HTTP ${response.status}`);
      }

      setUsers((currentUsers) =>
        sortUsers(
          currentUsers.filter((user) => String(user.id) !== idToDelete),
          sortBy
        )
      );
      setFeedback({
        type: 'success',
        text: data.message || `Deleted user ${idToDelete}.`,
      });
    } catch (error) {
      setFeedback({
        type: 'error',
        text: error.message,
      });
      throw error;
    }
  }

  async function createUser(nextUser) {
    const payload = {
      id: nextUser.id.trim(),
      first_name: nextUser.first_name.trim(),
      user_group: Number(nextUser.user_group),
    };

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await readJson(response);
    if (!response.ok) {
      const message = data.message || `Create failed: HTTP ${response.status}`;
      setFeedback({
        type: 'error',
        text: message,
      });
      throw new Error(message);
    }

    setUsers((currentUsers) => {
      const remainingUsers = currentUsers.filter(
        (user) => String(user.id) !== String(data.id)
      );
      return sortUsers([...remainingUsers, data], sortBy);
    });
    setFeedback({
      type: 'success',
      text: data.message || `Created user ${data.id}.`,
    });
  }

  async function updateUser(nextUser) {
    const userId = nextUser.id.trim();
    const payload = {
      first_name: nextUser.first_name.trim(),
      user_group: Number(nextUser.user_group),
    };

    const response = await fetch(`${API_URL}/${userId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data = await readJson(response);
    if (!response.ok) {
      const message = data.message || `Update failed: HTTP ${response.status}`;
      setFeedback({
        type: 'error',
        text: message,
      });
      throw new Error(message);
    }

    setUsers((currentUsers) => {
      const remainingUsers = currentUsers.filter(
        (user) => String(user.id) !== String(data.id)
      );
      return sortUsers([...remainingUsers, data], sortBy);
    });
    setFeedback({
      type: 'success',
      text: data.message || `Updated user ${data.id}.`,
    });
  }

  return (
    <>
      <section className="panel">
        <h1>User Directory</h1>
      </section>

      <section className="panel">
        <h2>Add or Update User</h2>
        <AddUserForm
          onCreate={createUser}
          onUpdate={updateUser}
        />
      </section>

      <section className="panel">
        <h2>Controls</h2>
        <Controls
          onDeleteClick={deleteUser}
          onSortByGroupClick={sortUsersByGroup}
          onSortByIdClick={sortUsersById}
          onViewToggleClick={toggleViewMode}
        />
      </section>

      {feedback ? (
        <section
          className={`panel status-panel ${
            feedback.type === 'error' ? 'status-panel-error' : 'status-panel-success'
          }`}
        >
          <p className="status" role="status">
            {feedback.text}
          </p>
        </section>
      ) : null}

      <section className="panel">
        <h2>All Users</h2>
        <UserList
          isLoading={isLoading}
          users={users}
          viewMode={viewMode}
        />
      </section>
    </>
  );
}

export default UserDirectoryPage;
