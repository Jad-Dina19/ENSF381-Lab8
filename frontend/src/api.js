export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5050';

export async function readJson(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new Error('The server returned an invalid JSON response.');
  }
}
