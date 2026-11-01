import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <section className="landing-hero landing-hero-solo">
      <div>
        <h1>Will this storm rapidly intensify?</h1>
        <p className="sub">
          Enter wind, location, and basin to get one clear chance of rapid intensification in the next 24 hours.
        </p>
        <div className="hero-ctas">
          <Link className="btn btn-white" to="/explorer">
            Predict RI
          </Link>
        </div>
      </div>
    </section>
  );
}
