import { useState } from 'react';
import './App.css';
import UserDirectoryPage from './exercise1/UserDirectoryPage';
import HousePricePredictorPage from './exercise2/HousePricePredictorPage';

function App() {
  const [currentPage, setCurrentPage] = useState('directory');

  return (
    <main className="app-shell">
      <section className="panel">
        <h1>ENSF-381 Lab 08</h1>
      </section>

      <section className="panel">
        <div className="nav-buttons">
          <button
            className={`btn ${currentPage === 'directory' ? 'btn-active' : ''}`}
            onClick={() => setCurrentPage('directory')}
            type="button"
          >
            Exercise 1: User Directory API
          </button>
          <button
            className={`btn ${currentPage === 'predictor' ? 'btn-active' : ''}`}
            onClick={() => setCurrentPage('predictor')}
            type="button"
          >
            Exercise 2: House Price Predictor
          </button>
        </div>
      </section>

      {currentPage === 'directory' ? <UserDirectoryPage /> : <HousePricePredictorPage />}
    </main>
  );
}

export default App;
