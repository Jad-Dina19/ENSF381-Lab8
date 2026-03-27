import { useState } from 'react';

const EMPTY_FORM = {
  id: '',
  first_name: '',
  user_group: '',
};

function AddUserForm({ onCreate, onUpdate }) {
  const [formData, setFormData] = useState(EMPTY_FORM);

  function handleChange(event) {
    const { name, value } = event.target;
    setFormData((currentFormData) => ({
      ...currentFormData,
      [name]: value,
    }));
  }

  async function submitForm(action) {
    try {
      await action(formData);
      setFormData(EMPTY_FORM);
    } catch {}
  }

  return (
    <form
      className="form-grid user-form"
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="form-field user-form-id-field">
        <label htmlFor="add-user-id">ID</label>
        <input
          id="add-user-id"
          min="1"
          name="id"
          onChange={handleChange}
          required
          type="number"
          value={formData.id}
        />
      </div>

      <div className="form-field user-form-name-field">
        <label htmlFor="add-user-first-name">First Name</label>
        <input
          id="add-user-first-name"
          name="first_name"
          onChange={handleChange}
          required
          type="text"
          value={formData.first_name}
        />
      </div>

      <div className="form-field user-form-group-field">
        <label htmlFor="add-user-group">User Group</label>
        <input
          id="add-user-group"
          min="1"
          name="user_group"
          onChange={handleChange}
          required
          type="number"
          value={formData.user_group}
        />
      </div>

      <button
        className="btn btn-active user-form-button"
        onClick={() => submitForm(onCreate)}
        type="button"
      >
        Add User
      </button>
      <button
        className="btn user-form-button"
        onClick={() => submitForm(onUpdate)}
        type="button"
      >
        Update User
      </button>
    </form>
  );
}

export default AddUserForm;
