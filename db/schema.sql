CREATE DATABASE IF NOT EXISTS claim_sentinel;
USE claim_sentinel;

CREATE TABLE policyholders (
    policy_number           INT PRIMARY KEY,
    months_as_customer      INT,
    age                     INT,
    policy_bind_date        DATE,
    policy_state            VARCHAR(10),
    policy_csl              VARCHAR(20),
    policy_deductable       INT,
    policy_annual_premium   DECIMAL(10,2),
    umbrella_limit          BIGINT,
    insured_zip             VARCHAR(10),
    insured_sex             VARCHAR(10),
    insured_education_level VARCHAR(50),
    insured_occupation      VARCHAR(50),
    insured_hobbies         VARCHAR(50),
    insured_relationship    VARCHAR(50),
    capital_gains           INT,
    capital_loss            INT
);

CREATE TABLE claims (
    claim_id                     VARCHAR(20) PRIMARY KEY,
    policy_number                INT,
    incident_date                DATE,
    incident_type                VARCHAR(50),
    collision_type                VARCHAR(50),
    incident_severity             VARCHAR(50),
    authorities_contacted         VARCHAR(50),
    incident_state                VARCHAR(10),
    incident_city                 VARCHAR(50),
    incident_location             VARCHAR(255),
    incident_hour_of_the_day      INT,
    number_of_vehicles_involved   INT,
    property_damage               VARCHAR(10),
    bodily_injuries                INT,
    witnesses                      INT,
    police_report_available        VARCHAR(10),
    total_claim_amount             DECIMAL(12,2),
    injury_claim                   DECIMAL(12,2),
    property_claim                 DECIMAL(12,2),
    vehicle_claim                  DECIMAL(12,2),
    auto_make                      VARCHAR(50),
    auto_model                     VARCHAR(50),
    auto_year                      INT,
    fraud_reported                 VARCHAR(5),         -- Y/N, ground-truth label
    fraud_probability              FLOAT DEFAULT NULL,  -- filled after ML scoring
    risk_flag                      VARCHAR(20) DEFAULT NULL,  -- low / high
    FOREIGN KEY (policy_number) REFERENCES policyholders(policy_number)
);

CREATE TABLE vehicle_check (
    auto_make        VARCHAR(50),
    auto_model       VARCHAR(50),
    auto_year        INT,
    model_exists      TINYINT(1),   -- NHTSA confirms this model exists for make+year
    checked_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (auto_make, auto_model, auto_year)
);

CREATE TABLE adjuster_decisions (
    decision_id     INT AUTO_INCREMENT PRIMARY KEY,
    claim_id        VARCHAR(20),
    decision        VARCHAR(20),        -- approved / escalated / denied
    reasoning       TEXT,               -- SHAP-based explanation text
    decided_by      VARCHAR(50),        -- 'agent' or adjuster name
    decided_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);

CREATE TABLE audit_log (
    log_id          INT AUTO_INCREMENT PRIMARY KEY,
    claim_id        VARCHAR(20),
    agent_name      VARCHAR(50),        -- IntakeAgent / FraudRiskAgent / AdjudicationAgent
    action          VARCHAR(100),
    details         TEXT,
    logged_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
);
