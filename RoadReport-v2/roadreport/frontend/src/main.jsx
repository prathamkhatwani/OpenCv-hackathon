import { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

class ErrorBoundary extends Component {
  state = { hasError: false, error: null };
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("UI Error caught by boundary:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '3rem', color: '#ffb45e', background: '#0e171b', minHeight: '100vh', fontFamily: 'system-ui, sans-serif' }}>
          <h2>Application Encountered an Interface Error</h2>
          <pre style={{ background: '#1c2830', padding: '1rem', borderRadius: '8px', overflow: 'auto', color: '#ff7b72' }}>
            {this.state.error?.message || String(this.state.error)}
          </pre>
          <button
            onClick={() => { this.setState({ hasError: false }); window.location.reload(); }}
            style={{ marginTop: '1rem', padding: '0.75rem 1.5rem', background: '#32d5b0', color: '#091014', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}
          >
            Reload Application
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)

