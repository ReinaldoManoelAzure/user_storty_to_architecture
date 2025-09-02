# Regras de Arquitetura Corporativa

## Segurança
- **SEC001 – Autenticação**: Todos os sistemas internos e externos devem usar OAuth2 ou OpenID Connect.
- **SEC002 – Criptografia**: Dados sensíveis devem ser criptografados em trânsito (HTTPS/TLS) e em repouso (AES-256).
- **SEC003 – Logs de Auditoria**: Toda integração deve gerar logs centralizados, armazenados por no mínimo 180 dias.

## Infraestrutura & Nuvem
- **INF001 – Cloud Provider**: Todas as soluções devem estar hospedadas em Azure.
- **INF002 – Deploy**: Microsserviços devem rodar em AKS (Kubernetes).
- **INF003 – Mensageria**: A fila padrão corporativa é Azure Service Bus.
- **INF004 – API Gateway**: Toda integração externa deve obrigatoriamente passar pelo Sensedia API Gateway.

## Observabilidade
- **OBS001 – Logging**: Todas as APIs devem enviar logs para o Azure Monitor.
- **OBS002 – Tracing**: Serviços críticos devem ter distributed tracing (OpenTelemetry) habilitado.
- **OBS003 – Métricas**: Deve-se coletar métricas de disponibilidade e latência em tempo real.

## Dados
- **DB001 – Banco Relacional**: O banco relacional padrão é SQL Server no Azure.
- **DB002 – Dados de Clientes**: Devem estar centralizados em IntegraFrota Database, nunca replicados em sistemas satélites.
- **DB003 – Backup**: Todo banco deve ter política de backup diário com retenção mínima de 30 dias.

## Integrações
- **INT001 – Protocolos**: Padrão primário é REST/JSON; gRPC só em casos internos com justificativa.
- **INT002 – Versionamento de APIs**: Toda API exposta deve ter versionamento explícito (v1, v2…).
- **INT003 – Resiliência**: Chamadas síncronas externas devem implementar retry com backoff exponencial.
