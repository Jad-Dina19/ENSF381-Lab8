function UserCard({ user }) {
  return (
    <article className="user-card">
      <h3>{user.first_name}</h3>
      <p>User Group: {user.user_group}</p>
      <p>ID: {user.id}</p>
    </article>
  );
}

export default UserCard;
