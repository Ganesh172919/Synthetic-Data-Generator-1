import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import Templates from './pages/Templates';
import DomainBuilder from './pages/DomainBuilder';
import Documentation from './pages/Documentation';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/templates" element={<Templates />} />
            <Route path="/domain-builder" element={<DomainBuilder />} />
            <Route path="/documentation" element={<Documentation />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
