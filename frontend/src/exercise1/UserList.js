import UserCard from './UserCard';

function UserList({ isLoading, users, viewMode }) {
  if (isLoading) {
    return <p>Loading users...</p>;
  }

  if (!users.length) {
    return <p>No users found.</p>;
  }

  return (
    <div className={viewMode === 'list' ? 'user-list' : 'user-grid'}>
      {users.map((user) => (
        <UserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

export default UserList;
