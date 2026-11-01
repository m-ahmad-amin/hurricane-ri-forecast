import { Link } from "react-router-dom";

export default function AboutPage() {
  return (
    <div className="about-page">
      <div className="about-hero">
        <h1>About PRIMED RI</h1>
        <p>
          A simple research tool that estimates the chance a tropical cyclone rapidly intensifies in the next 24
          hours.
        </p>
      </div>

      <div className="about-grid">
        <section className="about-card">
          <h2>What you get</h2>
          <p>
            Enter basin, wind speed, and location. The app returns one RI chance score in plain language: low,
            elevated, or high.
          </p>
        </section>

        <section className="about-card">
          <h2>How it works</h2>
          <p>
            Estimates are based on public TC PRIMED cases and similar past storms. The score is a calibrated
            probability from our combined model, not a full live satellite pipeline.
          </p>
        </section>

        <section className="about-card">
          <h2>Important limits</h2>
          <p>
            This is educational research software. It is not NHC or NWS guidance, and it is not a replacement for
            SHIPS-RII or official forecasts.
          </p>
        </section>
      </div>

      <div className="about-actions">
        <Link className="btn btn-navy" to="/explorer">
          Predict RI
        </Link>
        <Link className="btn btn-ghost-dark" to="/">
          Back to home
        </Link>
      </div>
    </div>
  );
}
