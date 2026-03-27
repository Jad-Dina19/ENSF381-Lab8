import { useState } from 'react';
import { API_BASE_URL, readJson } from '../api';

const PREDICTION_API_URL = `${API_BASE_URL}/predict_house_price`;

const INITIAL_FORM = {
  city: 'Calgary',
  province: 'Alberta',
  latitude: '51.0447',
  longitude: '-114.0719',
  lease_term: 'Long Term',
  type: 'Apartment',
  beds: '2',
  baths: '1',
  sq_feet: '850',
  furnishing: 'Unfurnished',
  smoking: 'Non-Smoking',
  pets: false,
};

const PROVINCES = [
  'Alberta',
  'British Columbia',
  'Manitoba',
  'New Brunswick',
  'Newfoundland and Labrador',
  'Northwest Territories',
  'Nova Scotia',
  'Ontario',
  'Quebec',
  'Saskatchewan',
];

const LEASE_TERMS = [
  '12 months',
  '6 months',
  'Long Term',
  'Negotiable',
  'Short Term',
  'months',
];

const PROPERTY_TYPES = [
  'Acreage',
  'Apartment',
  'Basement',
  'Condo Unit',
  'Duplex',
  'House',
  'Loft',
  'Main Floor',
  'Mobile',
  'Room For Rent',
  'Townhouse',
  'Vacation Home',
];

const FURNISHING_OPTIONS = [
  'Furnished',
  'Negotiable',
  'Unfurnished',
  'Unfurnished, Negotiable',
];

const SMOKING_OPTIONS = [
  'Negotiable',
  'Non-Smoking',
  'Smoke Free Building',
  'Smoking Allowed',
];

function formatCurrency(value) {
  return `$${Number(value).toFixed(2)}`;
}

function HousePricePredictorPage() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [feedback, setFeedback] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [predictedPrice, setPredictedPrice] = useState(null);

  function handleChange(event) {
    const { checked, name, type, value } = event.target;
    setFormData((currentFormData) => ({
      ...currentFormData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setFeedback(null);

    try {
      const response = await fetch(PREDICTION_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await readJson(response);
      if (!response.ok) {
        throw new Error(data.message || `Prediction failed: HTTP ${response.status}`);
      }

      setPredictedPrice(Number(data.predicted_price));
    } catch (error) {
      setPredictedPrice(null);
      setFeedback({
        type: 'error',
        text: error.message,
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <section className="panel">
        <h1>House Price Predictor</h1>
        <p className="page-intro">
          Exercise 2 includes a prebuilt prediction form. Your Lab 08 task is to
          complete the Flask API that loads the model and returns the prediction.
        </p>
      </section>

      <section className="panel">
        <h2>Prediction Form</h2>
        <form className="form-grid predictor-grid" onSubmit={handleSubmit}>
          <div className="form-field">
            <label htmlFor="prediction-city">City</label>
            <input
              id="prediction-city"
              name="city"
              onChange={handleChange}
              required
              type="text"
              value={formData.city}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-province">Province</label>
            <select
              id="prediction-province"
              name="province"
              onChange={handleChange}
              required
              value={formData.province}
            >
              {PROVINCES.map((province) => (
                <option key={province} value={province}>
                  {province}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="prediction-latitude">Latitude</label>
            <input
              id="prediction-latitude"
              name="latitude"
              onChange={handleChange}
              required
              step="any"
              type="number"
              value={formData.latitude}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-longitude">Longitude</label>
            <input
              id="prediction-longitude"
              name="longitude"
              onChange={handleChange}
              required
              step="any"
              type="number"
              value={formData.longitude}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-lease-term">Lease Term</label>
            <select
              id="prediction-lease-term"
              name="lease_term"
              onChange={handleChange}
              required
              value={formData.lease_term}
            >
              {LEASE_TERMS.map((leaseTerm) => (
                <option key={leaseTerm} value={leaseTerm}>
                  {leaseTerm}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="prediction-type">Type of House</label>
            <select
              id="prediction-type"
              name="type"
              onChange={handleChange}
              required
              value={formData.type}
            >
              {PROPERTY_TYPES.map((propertyType) => (
                <option key={propertyType} value={propertyType}>
                  {propertyType}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="prediction-beds">Number of Beds</label>
            <input
              id="prediction-beds"
              min="0"
              name="beds"
              onChange={handleChange}
              required
              step="any"
              type="number"
              value={formData.beds}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-baths">Number of Baths</label>
            <input
              id="prediction-baths"
              min="0"
              name="baths"
              onChange={handleChange}
              required
              step="any"
              type="number"
              value={formData.baths}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-square-feet">Square Feet</label>
            <input
              id="prediction-square-feet"
              min="0"
              name="sq_feet"
              onChange={handleChange}
              required
              step="any"
              type="number"
              value={formData.sq_feet}
            />
          </div>

          <div className="form-field">
            <label htmlFor="prediction-furnishing">Furnishing</label>
            <select
              id="prediction-furnishing"
              name="furnishing"
              onChange={handleChange}
              required
              value={formData.furnishing}
            >
              {FURNISHING_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="prediction-smoking">Smoking</label>
            <select
              id="prediction-smoking"
              name="smoking"
              onChange={handleChange}
              required
              value={formData.smoking}
            >
              {SMOKING_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field checkbox-field">
            <input
              checked={formData.pets}
              id="prediction-pets"
              name="pets"
              onChange={handleChange}
              type="checkbox"
            />
            <label htmlFor="prediction-pets">Pets in household</label>
          </div>

          <div className="predictor-actions">
            <button className="btn btn-active" disabled={isSubmitting} type="submit">
              {isSubmitting ? 'Predicting...' : 'Predict Rent'}
            </button>
          </div>
        </form>
      </section>

      {feedback ? (
        <section className="panel status-panel status-panel-error">
          <p className="status" role="status">
            {feedback.text}
          </p>
        </section>
      ) : null}

      {predictedPrice !== null ? (
        <section className="panel prediction-panel">
          <h2>Prediction Result</h2>
          <p className="prediction-value">
            Predicted Monthly Rent: {formatCurrency(predictedPrice)}
          </p>
        </section>
      ) : null}
    </>
  );
}

export default HousePricePredictorPage;
