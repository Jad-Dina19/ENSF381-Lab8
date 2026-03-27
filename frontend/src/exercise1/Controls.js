import { useState } from 'react';

function Controls({
  onDeleteClick,
  onSortByGroupClick,
  onSortByIdClick,
  onViewToggleClick,
}) {
  const [deleteId, setDeleteId] = useState('');

  async function handleDeleteClick() {
    try {
      await onDeleteClick(deleteId);
      setDeleteId('');
    } catch {}
  }

  return (
    <div className="controls-row">
      <div className="delete-controls">
        <label htmlFor="delete-id-input">Delete by ID</label>
        <input
          id="delete-id-input"
          type="number"
          value={deleteId}
          onChange={(event) => setDeleteId(event.target.value)}
        />
        <button
          className="btn btn-danger"
          onClick={handleDeleteClick}
        >
          Delete
        </button>
      </div>
      <div className="other-controls">
        <button className="btn" onClick={onSortByGroupClick}>
          Sort by Group
        </button>
        <button className="btn" onClick={onSortByIdClick}>
          Sort by ID
        </button>
        <button className="btn" onClick={onViewToggleClick}>
          Grid / List View
        </button>
      </div>
    </div>
  );
}

export default Controls;
