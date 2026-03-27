function AppButton({ label, onClick, variant = '' }) {
  const className = variant ? `btn btn-${variant}` : 'btn';

  return (
    <button className={className}  onClick={onClick}>
      {label}
    </button>
  );
}

export default AppButton;
