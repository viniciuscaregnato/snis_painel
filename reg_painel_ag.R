library(plm)
library(tidyverse)
library(glmnet)


# 1. Importação e filtragem

df_ag <- read.csv("df_ag.csv")

# Filtrar somente após 2010
df_ag <- df_ag %>% filter(Ano_Ref > 2010)

# Remover linhas com NA (necessário para glmnet e para plm ficar consistente)
df_ag <- na.omit(df_ag)


# 2. Preparar painel

df_ag_painel <- pdata.frame(df_ag, index = c("id_mun", "Ano_Ref"))

# 3. Preparar dados para LASSO
#    (Usando transformação "within" – CORRETO para painel)


# fórmula completa
variaveis_preditoras <- setdiff(names(df_ag), c("AG001", "id_mun", "Ano_Ref"))
form_all <- as.formula(paste("AG001 ~", paste(variaveis_preditoras, collapse = " + ")))

# Rodar modelo FE para obter transformações "within"
modelo_fe_temp <- plm(form_all, data = df_ag_painel, model = "within")

# Extrair y e X transformados
y_fe <- pmodel.response(modelo_fe_temp)
X_fe <- model.matrix(modelo_fe_temp)[, -1]  # remove intercepto inexistente no within


# 4. Rodar LASSO via validação cruzada

cv_lasso <- cv.glmnet(X_fe, y_fe, alpha = 1)

# Coeficientes selecionados (diferentes de zero)
coef_lasso <- coef(cv_lasso, s = "lambda.min")
vars_selecionadas <- rownames(coef_lasso)[coef_lasso[,1] != 0]

# Remover o intercepto, se aparecer
vars_selecionadas <- vars_selecionadas[vars_selecionadas != "(Intercept)"]

cat("Variáveis selecionadas pelo LASSO:\n")
print(vars_selecionadas)


# 5. Criar fórmula final com variáveis selecionadas

formula_final <- as.formula(
  paste("AG001 ~", paste(vars_selecionadas, collapse = " + "))
)

cat("\nFórmula final do modelo:\n")
print(formula_final)


# 6. Rodar modelos FE e RE com variáveis selecionadas

modelo_FE_final <- plm(formula_final, data = df_ag_painel, model = "within")
modelo_RE_final <- plm(formula_final, data = df_ag_painel, model = "random")


# 7. Teste de Hausman

haus <- phtest(modelo_FE_final, modelo_RE_final)
print(haus)


# 8. Indicar modelo adequado

if(haus$p.value < 0.05){
  cat("\n➡️ O teste de Hausman rejeita H0 → Use **Efeitos Fixos (FE)**\n")
  modelo_final <- modelo_FE_final
} else {
  cat("\n➡️ O teste de Hausman NÃO rejeita H0 → Use **Efeitos Aleatórios (RE)**\n")
  modelo_final <- modelo_RE_final
}

cat("\nResumo do modelo final escolhido:\n")
summary(modelo_final)
