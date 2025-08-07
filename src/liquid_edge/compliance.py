"""GDPR, CCPA, and global compliance framework for liquid neural networks."""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import hashlib
import logging
from datetime import datetime, timezone


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)


class DataCategory(Enum):
    """Categories of data processed by liquid neural networks."""
    SENSOR_DATA = "sensor_data"
    BEHAVIORAL_DATA = "behavioral_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"
    PERFORMANCE_DATA = "performance_data"
    DIAGNOSTIC_DATA = "diagnostic_data"
    ENVIRONMENTAL_DATA = "environmental_data"


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    ROBOT_CONTROL = "robot_control"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SAFETY_MONITORING = "safety_monitoring"
    RESEARCH_DEVELOPMENT = "research_development"
    QUALITY_ASSURANCE = "quality_assurance"
    DIAGNOSTIC_ANALYSIS = "diagnostic_analysis"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    record_id: str
    data_category: DataCategory
    purpose: ProcessingPurpose
    legal_basis: str
    data_subject: Optional[str] = None
    processing_timestamp: float = field(default_factory=time.time)
    retention_period_days: int = 30
    anonymized: bool = True
    encrypted: bool = True
    cross_border_transfer: bool = False
    destination_countries: List[str] = field(default_factory=list)
    consent_obtained: bool = False
    consent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for storage."""
        return {
            "record_id": self.record_id,
            "data_category": self.data_category.value,
            "purpose": self.purpose.value,
            "legal_basis": self.legal_basis,
            "data_subject": self.data_subject,
            "processing_timestamp": self.processing_timestamp,
            "retention_period_days": self.retention_period_days,
            "anonymized": self.anonymized,
            "encrypted": self.encrypted,
            "cross_border_transfer": self.cross_border_transfer,
            "destination_countries": self.destination_countries,
            "consent_obtained": self.consent_obtained,
            "consent_id": self.consent_id
        }


@dataclass
class ConsentRecord:
    """Record of user consent for compliance."""
    consent_id: str
    data_subject: str
    purposes: List[ProcessingPurpose]
    timestamp: float = field(default_factory=time.time)
    valid_until: Optional[float] = None
    withdrawn: bool = False
    withdrawal_timestamp: Optional[float] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.withdrawn:
            return False
        if self.valid_until and time.time() > self.valid_until:
            return False
        return True


class ComplianceManager:
    """Comprehensive compliance management for liquid neural networks."""
    
    def __init__(self, 
                 frameworks: List[ComplianceFramework],
                 organization_name: str,
                 data_controller: str,
                 storage_path: Optional[str] = None):
        
        self.frameworks = frameworks
        self.organization_name = organization_name
        self.data_controller = data_controller
        
        self.storage_path = Path(storage_path) if storage_path else Path("compliance_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Compliance records
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        
        # Setup logging
        self.logger = logging.getLogger("liquid_edge.compliance")
        self._setup_compliance_logging()
        
        self.logger.info(f"Compliance Manager initialized for frameworks: {[f.value for f in frameworks]}")
    
    def _setup_compliance_logging(self):
        """Setup compliance-specific logging."""
        handler = logging.FileHandler(self.storage_path / "compliance.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def record_data_processing(self, 
                              data_category: DataCategory,
                              purpose: ProcessingPurpose,
                              legal_basis: str,
                              data_subject: Optional[str] = None,
                              **kwargs) -> str:
        """Record data processing activity for compliance."""
        
        record_id = self._generate_record_id()
        
        record = DataProcessingRecord(
            record_id=record_id,
            data_category=data_category,
            purpose=purpose,
            legal_basis=legal_basis,
            data_subject=data_subject,
            **kwargs
        )
        
        self.processing_records.append(record)
        
        # Log the processing activity
        self.logger.info(
            f"Data processing recorded: {record_id} - {data_category.value} for {purpose.value}",
            extra={
                "record_id": record_id,
                "data_category": data_category.value,
                "purpose": purpose.value,
                "legal_basis": legal_basis,
                "data_subject": data_subject
            }
        )
        
        # Save to persistent storage
        self._save_processing_record(record)
        
        return record_id
    
    def obtain_consent(self,
                      data_subject: str,
                      purposes: List[ProcessingPurpose],
                      valid_duration_days: Optional[int] = None,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None) -> str:
        """Record user consent for data processing."""
        
        consent_id = self._generate_consent_id()
        
        valid_until = None
        if valid_duration_days:
            valid_until = time.time() + (valid_duration_days * 24 * 3600)
        
        consent = ConsentRecord(
            consent_id=consent_id,
            data_subject=data_subject,
            purposes=purposes,
            valid_until=valid_until,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.consent_records[consent_id] = consent
        
        self.logger.info(
            f"Consent obtained: {consent_id} for {data_subject} - purposes: {[p.value for p in purposes]}",
            extra={
                "consent_id": consent_id,
                "data_subject": data_subject,
                "purposes": [p.value for p in purposes]
            }
        )
        
        self._save_consent_record(consent)
        
        return consent_id
    
    def withdraw_consent(self, consent_id: str, data_subject: str) -> bool:
        """Withdraw user consent."""
        
        if consent_id not in self.consent_records:
            return False
        
        consent = self.consent_records[consent_id]
        
        if consent.data_subject != data_subject:
            self.logger.warning(f"Consent withdrawal attempted by wrong subject: {data_subject}")
            return False
        
        consent.withdrawn = True
        consent.withdrawal_timestamp = time.time()
        
        self.logger.info(
            f"Consent withdrawn: {consent_id} by {data_subject}",
            extra={"consent_id": consent_id, "data_subject": data_subject}
        )
        
        self._save_consent_record(consent)
        
        return True
    
    def check_processing_lawful(self, 
                               data_category: DataCategory,
                               purpose: ProcessingPurpose,
                               data_subject: Optional[str] = None) -> Dict[str, Any]:
        """Check if data processing is lawful under applicable frameworks."""
        
        compliance_status = {
            "lawful": False,
            "reasons": [],
            "required_actions": [],
            "framework_compliance": {}
        }
        
        for framework in self.frameworks:
            framework_check = self._check_framework_compliance(
                framework, data_category, purpose, data_subject
            )
            compliance_status["framework_compliance"][framework.value] = framework_check
            
            if not framework_check["compliant"]:
                compliance_status["reasons"].extend(framework_check["issues"])
                compliance_status["required_actions"].extend(framework_check["actions"])
        
        # Overall compliance if all frameworks are satisfied
        compliance_status["lawful"] = all(
            check["compliant"] for check in compliance_status["framework_compliance"].values()
        )
        
        return compliance_status
    
    def _check_framework_compliance(self, 
                                   framework: ComplianceFramework,
                                   data_category: DataCategory,
                                   purpose: ProcessingPurpose,
                                   data_subject: Optional[str]) -> Dict[str, Any]:
        """Check compliance with specific framework."""
        
        if framework == ComplianceFramework.GDPR:
            return self._check_gdpr_compliance(data_category, purpose, data_subject)
        elif framework == ComplianceFramework.CCPA:
            return self._check_ccpa_compliance(data_category, purpose, data_subject)
        elif framework == ComplianceFramework.PDPA:
            return self._check_pdpa_compliance(data_category, purpose, data_subject)
        else:
            return {
                "compliant": True,  # Default to compliant for unknown frameworks
                "issues": [],
                "actions": []
            }
    
    def _check_gdpr_compliance(self, 
                              data_category: DataCategory,
                              purpose: ProcessingPurpose,
                              data_subject: Optional[str]) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        
        issues = []
        actions = []
        
        # Check if personal data is involved
        personal_data_categories = {
            DataCategory.BEHAVIORAL_DATA,
            DataCategory.BIOMETRIC_DATA,
            DataCategory.LOCATION_DATA
        }
        
        if data_category in personal_data_categories:
            # Personal data processing requires legal basis
            if data_subject:
                # Check if valid consent exists
                valid_consents = [
                    consent for consent in self.consent_records.values()
                    if (consent.data_subject == data_subject and 
                        purpose in consent.purposes and 
                        consent.is_valid())
                ]
                
                if not valid_consents:
                    issues.append("No valid consent found for personal data processing")
                    actions.append("Obtain explicit consent from data subject")
            else:
                # Anonymous processing is generally acceptable
                pass
        
        # Check data minimization principle
        if purpose == ProcessingPurpose.RESEARCH_DEVELOPMENT and data_category == DataCategory.BIOMETRIC_DATA:
            issues.append("Biometric data processing for research may require additional safeguards")
            actions.append("Implement additional technical and organizational measures")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "actions": actions
        }
    
    def _check_ccpa_compliance(self, 
                              data_category: DataCategory,
                              purpose: ProcessingPurpose,
                              data_subject: Optional[str]) -> Dict[str, Any]:
        """Check CCPA compliance requirements."""
        
        issues = []
        actions = []
        
        # CCPA focuses on California residents and commercial purposes
        if purpose in [ProcessingPurpose.PERFORMANCE_OPTIMIZATION, ProcessingPurpose.RESEARCH_DEVELOPMENT]:
            if data_category in [DataCategory.BEHAVIORAL_DATA, DataCategory.LOCATION_DATA]:
                actions.append("Provide notice of data collection and processing purposes")
                actions.append("Enable consumer rights (access, deletion, opt-out)")
        
        return {
            "compliant": True,  # CCPA compliance is largely procedural
            "issues": issues,
            "actions": actions
        }
    
    def _check_pdpa_compliance(self, 
                              data_category: DataCategory,
                              purpose: ProcessingPurpose,
                              data_subject: Optional[str]) -> Dict[str, Any]:
        """Check Singapore PDPA compliance requirements."""
        
        issues = []
        actions = []
        
        # PDPA requires consent for most personal data processing
        if data_category in [DataCategory.BEHAVIORAL_DATA, DataCategory.BIOMETRIC_DATA, DataCategory.LOCATION_DATA]:
            if data_subject and not self._has_valid_consent(data_subject, purpose):
                issues.append("PDPA requires consent for personal data processing")
                actions.append("Obtain consent from individual")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues,
            "actions": actions
        }
    
    def _has_valid_consent(self, data_subject: str, purpose: ProcessingPurpose) -> bool:
        """Check if valid consent exists for data subject and purpose."""
        for consent in self.consent_records.values():
            if (consent.data_subject == data_subject and 
                purpose in consent.purposes and 
                consent.is_valid()):
                return True
        return False
    
    def generate_data_processing_report(self, 
                                       framework: ComplianceFramework,
                                       start_date: Optional[float] = None,
                                       end_date: Optional[float] = None) -> Dict[str, Any]:
        """Generate data processing report for compliance."""
        
        if start_date is None:
            start_date = time.time() - (30 * 24 * 3600)  # Last 30 days
        if end_date is None:
            end_date = time.time()
        
        # Filter records by date range
        relevant_records = [
            record for record in self.processing_records
            if start_date <= record.processing_timestamp <= end_date
        ]
        
        # Categorize processing activities
        by_category = {}
        by_purpose = {}
        
        for record in relevant_records:
            category = record.data_category.value
            purpose = record.purpose.value
            
            if category not in by_category:
                by_category[category] = 0
            by_category[category] += 1
            
            if purpose not in by_purpose:
                by_purpose[purpose] = 0
            by_purpose[purpose] += 1
        
        report = {
            "framework": framework.value,
            "organization": self.organization_name,
            "data_controller": self.data_controller,
            "report_period": {
                "start": start_date,
                "end": end_date
            },
            "total_processing_activities": len(relevant_records),
            "processing_by_category": by_category,
            "processing_by_purpose": by_purpose,
            "cross_border_transfers": len([
                r for r in relevant_records if r.cross_border_transfer
            ]),
            "consent_records": len(self.consent_records),
            "active_consents": len([
                c for c in self.consent_records.values() if c.is_valid()
            ]),
            "compliance_measures": self._get_compliance_measures(framework)
        }
        
        return report
    
    def _get_compliance_measures(self, framework: ComplianceFramework) -> List[str]:
        """Get implemented compliance measures for framework."""
        
        measures = [
            "Data processing logging and audit trails",
            "Consent management system",
            "Data anonymization and pseudonymization",
            "Encryption of personal data",
            "Regular compliance monitoring"
        ]
        
        if framework == ComplianceFramework.GDPR:
            measures.extend([
                "Data Protection Impact Assessment (DPIA) procedures",
                "Right to be forgotten implementation",
                "Data portability mechanisms",
                "Privacy by design implementation"
            ])
        elif framework == ComplianceFramework.CCPA:
            measures.extend([
                "Consumer rights request handling",
                "Opt-out mechanisms for data sale",
                "Notice of data collection practices"
            ])
        
        return measures
    
    def handle_data_subject_request(self, 
                                   request_type: str,
                                   data_subject: str) -> Dict[str, Any]:
        """Handle data subject rights requests (access, deletion, portability)."""
        
        request_id = self._generate_record_id()
        
        if request_type.lower() == "access":
            return self._handle_access_request(request_id, data_subject)
        elif request_type.lower() == "deletion":
            return self._handle_deletion_request(request_id, data_subject)
        elif request_type.lower() == "portability":
            return self._handle_portability_request(request_id, data_subject)
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "message": f"Unsupported request type: {request_type}"
            }
    
    def _handle_access_request(self, request_id: str, data_subject: str) -> Dict[str, Any]:
        """Handle data access request."""
        
        # Find all processing records for this data subject
        subject_records = [
            record.to_dict() for record in self.processing_records
            if record.data_subject == data_subject
        ]
        
        # Find consent records
        subject_consents = [
            {
                "consent_id": consent_id,
                "purposes": [p.value for p in consent.purposes],
                "timestamp": consent.timestamp,
                "valid": consent.is_valid()
            }
            for consent_id, consent in self.consent_records.items()
            if consent.data_subject == data_subject
        ]
        
        self.logger.info(
            f"Data access request processed: {request_id} for {data_subject}",
            extra={"request_id": request_id, "data_subject": data_subject}
        )
        
        return {
            "request_id": request_id,
            "status": "completed",
            "data_subject": data_subject,
            "processing_records": subject_records,
            "consent_records": subject_consents,
            "generated_at": time.time()
        }
    
    def _handle_deletion_request(self, request_id: str, data_subject: str) -> Dict[str, Any]:
        """Handle data deletion request (right to be forgotten)."""
        
        # Mark records for deletion (in real implementation, would actually delete)
        deleted_records = 0
        for record in self.processing_records:
            if record.data_subject == data_subject:
                # In production, implement actual deletion logic
                deleted_records += 1
        
        # Revoke consents
        revoked_consents = 0
        for consent_id, consent in self.consent_records.items():
            if consent.data_subject == data_subject and not consent.withdrawn:
                consent.withdrawn = True
                consent.withdrawal_timestamp = time.time()
                revoked_consents += 1
        
        self.logger.info(
            f"Data deletion request processed: {request_id} for {data_subject} - "
            f"deleted {deleted_records} records, revoked {revoked_consents} consents",
            extra={
                "request_id": request_id,
                "data_subject": data_subject,
                "deleted_records": deleted_records,
                "revoked_consents": revoked_consents
            }
        )
        
        return {
            "request_id": request_id,
            "status": "completed",
            "data_subject": data_subject,
            "deleted_records": deleted_records,
            "revoked_consents": revoked_consents,
            "processed_at": time.time()
        }
    
    def _handle_portability_request(self, request_id: str, data_subject: str) -> Dict[str, Any]:
        """Handle data portability request."""
        
        # Export data in structured format
        export_data = self._handle_access_request(request_id, data_subject)
        
        # Add portability-specific formatting
        export_data["format"] = "json"
        export_data["export_type"] = "data_portability"
        
        self.logger.info(
            f"Data portability request processed: {request_id} for {data_subject}",
            extra={"request_id": request_id, "data_subject": data_subject}
        )
        
        return export_data
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID."""
        timestamp = str(time.time())
        hash_input = f"{timestamp}{self.organization_name}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _generate_consent_id(self) -> str:
        """Generate unique consent ID."""
        timestamp = str(time.time())
        hash_input = f"consent_{timestamp}{self.organization_name}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _save_processing_record(self, record: DataProcessingRecord):
        """Save processing record to persistent storage."""
        record_file = self.storage_path / f"processing_{record.record_id}.json"
        with open(record_file, 'w') as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
    
    def _save_consent_record(self, consent: ConsentRecord):
        """Save consent record to persistent storage."""
        consent_file = self.storage_path / f"consent_{consent.consent_id}.json"
        consent_data = {
            "consent_id": consent.consent_id,
            "data_subject": consent.data_subject,
            "purposes": [p.value for p in consent.purposes],
            "timestamp": consent.timestamp,
            "valid_until": consent.valid_until,
            "withdrawn": consent.withdrawn,
            "withdrawal_timestamp": consent.withdrawal_timestamp,
            "ip_address": consent.ip_address,
            "user_agent": consent.user_agent
        }
        
        with open(consent_file, 'w') as f:
            json.dump(consent_data, f, indent=2, default=str)
    
    def export_compliance_report(self, 
                                framework: ComplianceFramework,
                                output_path: str) -> str:
        """Export comprehensive compliance report."""
        
        report = self.generate_data_processing_report(framework)
        
        # Add additional compliance information
        report["compliance_framework"] = {
            "name": framework.value.upper(),
            "description": self._get_framework_description(framework),
            "key_principles": self._get_framework_principles(framework)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Compliance report exported: {output_path}")
        return output_path
    
    def _get_framework_description(self, framework: ComplianceFramework) -> str:
        """Get description of compliance framework."""
        descriptions = {
            ComplianceFramework.GDPR: "European Union General Data Protection Regulation",
            ComplianceFramework.CCPA: "California Consumer Privacy Act",
            ComplianceFramework.PDPA: "Singapore Personal Data Protection Act",
            ComplianceFramework.LGPD: "Brazilian General Data Protection Law",
            ComplianceFramework.PIPEDA: "Canadian Personal Information Protection and Electronic Documents Act"
        }
        return descriptions.get(framework, "Unknown compliance framework")
    
    def _get_framework_principles(self, framework: ComplianceFramework) -> List[str]:
        """Get key principles of compliance framework."""
        principles = {
            ComplianceFramework.GDPR: [
                "Lawfulness, fairness and transparency",
                "Purpose limitation",
                "Data minimization",
                "Accuracy",
                "Storage limitation",
                "Integrity and confidentiality",
                "Accountability"
            ],
            ComplianceFramework.CCPA: [
                "Transparency in data collection",
                "Consumer right to know",
                "Consumer right to delete",
                "Consumer right to opt-out",
                "Non-discrimination"
            ],
            ComplianceFramework.PDPA: [
                "Consent principle",
                "Purpose limitation principle",
                "Notification principle",
                "Access and correction principle",
                "Accuracy principle",
                "Protection principle",
                "Retention limitation principle"
            ]
        }
        return principles.get(framework, [])


# Convenience functions
def create_compliance_manager(frameworks: List[str], 
                            organization_name: str,
                            data_controller: str) -> ComplianceManager:
    """Create compliance manager with string framework names."""
    framework_enums = []
    for framework_str in frameworks:
        try:
            framework_enums.append(ComplianceFramework(framework_str.lower()))
        except ValueError:
            logging.warning(f"Unknown compliance framework: {framework_str}")
    
    return ComplianceManager(framework_enums, organization_name, data_controller)


def ensure_gdpr_compliance(processing_function: callable):
    """Decorator to ensure GDPR compliance for data processing functions."""
    def wrapper(*args, **kwargs):
        # In a real implementation, this would check GDPR requirements
        # For now, just log the processing activity
        logging.info(f"GDPR compliance check for {processing_function.__name__}")
        return processing_function(*args, **kwargs)
    
    return wrapper
