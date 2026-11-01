import { NavLink, Outlet, Route, Routes } from "react-router-dom";
import LogoMark from "./components/LogoMark";
import AboutPage from "./pages/AboutPage";
import CasePage from "./pages/CasePage";
import ExplorerPage from "./pages/ExplorerPage";
import HomePage from "./pages/HomePage";

function LandingLayout() {
  return (
    <div className="landing">
      <div className="landing-bg" />
      <header className="landing-top">
        <NavLink to="/" className="logo" end>
          <LogoMark />
          PRIMED RI
        </NavLink>
        <nav className="pill-nav">
          <NavLink to="/explorer">Predict</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <div className="top-actions">
          <a className="btn btn-ghost" href="https://github.com/m-ahmad-amin/hurricane-ri-forecast" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <NavLink className="btn btn-white" to="/explorer">
            Predict RI
          </NavLink>
        </div>
      </header>
      <Outlet />
      <p className="landing-note">
        Research tool · Not official NHC guidance
      </p>
    </div>
  );
}

function AppLayout() {
  return (
    <div className="app-shell">
      <header className="app-bar">
        <NavLink to="/" className="app-logo">
          <LogoMark className="dark" />
          PRIMED RI
        </NavLink>
        <nav className="app-tabs">
          <NavLink to="/explorer">Predict</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <p className="app-hint">Enter storm details to get RI chance</p>
      </header>
      <main className="app-body">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<LandingLayout />}>
        <Route path="/" element={<HomePage />} />
      </Route>
      <Route element={<AppLayout />}>
        <Route path="/explorer" element={<ExplorerPage />} />
        <Route path="/cases/:caseId" element={<CasePage />} />
        <Route path="/about" element={<AboutPage />} />
      </Route>
    </Routes>
  );
}
